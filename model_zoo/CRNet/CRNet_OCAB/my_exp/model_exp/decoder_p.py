import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
import cv2


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 可学习参数
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
    def initialize(self):
        weight_init(self)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)  # 128*4
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):  # 2, 128, 24, 24
        x = self.project_in(x)  # 2, 1024, 24, 24
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # x1: 2, 512, 24, 24; x2: 2, 512, 24, 24 深度可分离卷积
        x = F.gelu(x1) * x2  # 2, 512, 24, 24
        x = self.project_out(x)  # 2, 128, 24, 24
        return x

    def initialize(self):
        weight_init(self)


def get_open_map(input,kernel_size,iterations):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    open_map_list = map(lambda i: cv2.dilate(i.permute(1, 2, 0).detach().numpy(), kernel=kernel, iterations=iterations), input.cpu())
    open_map_tensor = torch.from_numpy(np.array(list(open_map_list)))
    return open_map_tensor.unsqueeze(1).cuda()


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, -1, h, w)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, window_size, overlap_ratio, bias):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)

        # self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        #
        # self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        # self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.mask_ratio = torch.tensor(0.3)
        # self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x, mask, rpi):
        b,c,h,w = x.shape  # 1, 64, 24, 24
        x = x.view(b, h, w, c)

        # q=self.qkv1conv(self.qkv_0(x))  # 1, 64, 24, 24
        # k=self.qkv2conv(self.qkv_1(x))  # 1, 64, 24, 24
        # v=self.qkv3conv(self.qkv_2(x))  # 1, 64, 24, 24

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)  # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1)  # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1)  # b, 2*c, h, w

        if mask is not None:
            mask = mask.permute(0, 2, 3, 1)
            q=q*mask  # 1, 64, 24, 24
            # k=k*mask  # 1, 64, 24, 24

        # partition windows
        q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv)  # b, c*w*w, nw
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c,
                               owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()  # 2, nw*b, ow*ow, c
        k_windows, v_windows = kv_windows[0], kv_windows[1]  # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape  # b_: 16, nq: 25
        _, n, _ = k_windows.shape  # n: 49
        d = self.dim // self.num_heads  # d: 8
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, nq, d 16,8,25,8
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, n, d 16,8,49,8
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, n, d 16,8,49,8

        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 1, 8, 8, 576
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 1, 8, 8, 576
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 1, 8, 8, 576

        q = torch.nn.functional.normalize(q, dim=-1)  # 1, 8, 8, 576  最后一个维度进行归一化
        k = torch.nn.functional.normalize(k, dim=-1)  # 1, 8, 8, 576
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 1, 8, 8, 8

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size,
            -1)  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        # DropKey
        m_r = torch.ones_like(attn) * self.mask_ratio
        attn = attn + torch.bernoulli(m_r) * -1e12

        attn = attn.softmax(dim=-1)

        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        out = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        # out = out.view(b, h * w, self.dim)
        out = self.project_out(out)

        # out = (attn @ v)  # 1, 8, 8, 576
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)  # 1, 64, 24, 24
        # out = self.project_out(out)  # 1, 64, 24, 24
        # # 缩放参数
        # out = self.gamma * out
        return out

    def initialize(self):
        weight_init(self)


class MSA_head(nn.Module):
    def __init__(self, dim=128, num_heads=8, window_size=4, overlap_ratio=0.5, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        # relative position index
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, window_size, overlap_ratio, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]   # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def forward(self, x, mask=None):
        # params = {'rpi_oca': self.relative_position_index_OCA}
        x = x + self.attn(self.norm1(x), mask, self.relative_position_index_OCA)
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)


class MSA_module(nn.Module):
    def __init__(self, channel1, channel2, opr_kernel_size=3,iterations=1):
        super(MSA_module, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2
        self.opr_kernel_size = opr_kernel_size
        self.iterations = iterations
        dim = channel1
        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.B_TA = MSA_head(dim=dim)
        self.F_TA = MSA_head(dim=dim)
        self.TA = MSA_head(dim=dim)
        self.Fuse = nn.Conv2d(3*dim,dim,kernel_size=3,padding=1)
        self.Fuse2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

        # self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

    def forward(self, x, side_x, mask):
        """

        Args:
            x: 1, 64, 12, 12
            side_x: 1, 64, 24, 24
            mask: 1, 1, 12, 12

        Returns:

        """

        x = self.up(x)  # 1,64,24,24
        # print(x.size())
        N,C,H,W = x.shape  # C: 64, H: 24, N: 1, W: 24
        mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear')  # 1, 1, 24, 24
        mask_d = mask.detach()  # 1, 1, 24, 24
        mask_d = torch.sigmoid(mask_d)
        edge = get_open_map(mask_d, self.opr_kernel_size, self.iterations) - mask_d

        xf = self.F_TA(x, edge)  # 1,64,24,24
        xb = self.B_TA(x, 1-edge)  # 1,64,24,24
        x = self.TA(x, side_x)  # 1,64,24,24
        x = torch.cat((xb, xf, x),1)  # 1,192,24,24
        x = x.view(N,3*C,H,W)  # 1,192,24,24
        x = self.Fuse(x)  # 1,64,24,24
        D = self.Fuse2(side_x+side_x*x)  # 1,64,24,24

        # output_map = self.output_map(D)  # 1, 1, 24, 24

        # return D, output_map
        return D
    
    def initialize(self):
        weight_init(self)

class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels*3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels*2)

        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, input1, input2, input3):
        fuse = torch.cat((input1, input2, input3), 1)  # 2, 384, 24, 24
        fuse = self.bn1(self.conv1(fuse))  # 2, 128, 24, 24
        fuse = self.bn2(self.conv2(fuse))  # 2, 256, 24, 24
        fuse = self.bn3(self.conv3(fuse))  # 2, 128, 24, 24
        return fuse

    def initialize(self):
        weight_init(self)


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()

        self.side_conv1 = nn.Conv2d(512, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2 = nn.Conv2d(320, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3 = nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4 = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)

        self.conv_block = Conv_Block(channels)

        self.fuse1 = nn.Sequential(nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(channels))
        self.fuse2 = nn.Sequential(nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(channels))
        self.fuse3 = nn.Sequential(nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(channels))
       
        self.MSA5=MSA_module(dim = channels)
        self.MSA4=MSA_module(dim = channels)
        self.MSA3=MSA_module(dim = channels)
        self.MSA2=MSA_module(dim = channels)

        self.predtrans1  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans2  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans3  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans4  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans5  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

        self.initialize()
        


    def forward(self, E4, E3, E2, E1,shape):
        E4, E3, E2, E1= self.side_conv1(E4), self.side_conv2(E3), self.side_conv3(E2), self.side_conv4(E1)
        """
        E1: 2, 128, 96, 96
        E2: 2, 128, 48, 48
        E3: 2, 128, 24, 24
        E4: 2, 128, 12, 12
        """
        if E4.size()[2:] != E3.size()[2:]:
            E4 = F.interpolate(E4, size=E3.size()[2:], mode='bilinear')
        if E2.size()[2:] != E3.size()[2:]:
            E2 = F.interpolate(E2, size=E3.size()[2:], mode='bilinear')
        """
        E1: 2, 128, 96, 96
        E2: 2, 128, 24, 24
        E3: 2, 128, 24, 24
        E4: 2, 128, 24, 24
        """

        E5 = self.conv_block(E4, E3, E2)  # 2, 128, 24, 24

        E4 = torch.cat((E4, E5),1)  # 2, 256, 24, 24
        E3 = torch.cat((E3, E5),1)  # 2, 256, 24, 24
        E2 = torch.cat((E2, E5),1)  # 2, 256, 24, 24

        E4 = F.relu(self.fuse1(E4), inplace=True)  # 2, 128, 24, 24
        E3 = F.relu(self.fuse2(E3), inplace=True)  # 2, 128, 24, 24
        E2 = F.relu(self.fuse3(E2), inplace=True)  # 2, 128, 24, 24

        P5 = self.predtrans5(E5)  # 2, 1, 24, 24

        D4 = self.MSA5(E5, E4, P5)  # 2, 128, 24, 24
        D4 = F.interpolate(D4, size=E3.size()[2:], mode='bilinear')  # 2, 128, 24, 24
        P4  = self.predtrans4(D4)  # 2, 1, 24, 24
        
        D3 = self.MSA4(D4, E3, P4)  # 2, 128, 24, 24
        D3 = F.interpolate(D3,   size=E2.size()[2:], mode='bilinear')  # 2, 128, 24, 24
        P3  = self.predtrans3(D3)   # 2, 1, 24, 24
        
        D2 = self.MSA3(D3, E2, P3)  # 2, 128, 24, 24
        D2 = F.interpolate(D2, size=E1.size()[2:], mode='bilinear')  # 2, 128, 96, 96
        P2  = self.predtrans2(D2)  # 2, 1, 96, 96
        
        D1 = self.MSA2(D2, E1, P2)  # 2, 128, 96, 96
        P1  =self.predtrans1(D1)  # 2, 1, 96, 96

        P1 = F.interpolate(P1, size=shape, mode='bilinear')  # 2, 1, 384, 384
        P2 = F.interpolate(P2, size=shape, mode='bilinear')  # 2, 1, 384, 384
        P3 = F.interpolate(P3, size=shape, mode='bilinear')  # 2, 1, 384, 384
        P4 = F.interpolate(P4, size=shape, mode='bilinear')  # 2, 1, 384, 384
        P5 = F.interpolate(P5, size=shape, mode='bilinear')  # 2, 1, 384, 384
        
        return P5, P4, P3, P2, P1

    def initialize(self):
        weight_init(self)

 
