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
        elif isinstance(m, (
        nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d,
        nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
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
        hidden_features = int(dim * ffn_expansion_factor)  # 128*4
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):  # 2, 128, 24, 24
        x = self.project_in(x)  # 2, 1024, 24, 24
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # x1: 2, 512, 24, 24; x2: 2, 512, 24, 24 深度可分离卷积
        x = F.gelu(x1) * x2  # 2, 512, 24, 24
        x = self.project_out(x)  # 2, 128, 24, 24
        return x

    def initialize(self):
        weight_init(self)


def get_open_map(input, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    open_map_list = map(lambda i: cv2.dilate(i.permute(1, 2, 0).detach().numpy(), kernel=kernel, iterations=iterations),
                        input.cpu())
    open_map_tensor = torch.from_numpy(np.array(list(open_map_list)))
    return open_map_tensor.unsqueeze(1).cuda()


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode, opr_kernel_size=3, iterations=1):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.mask_ratio = torch.tensor(0.3)

        self.opr_kernel_size = opr_kernel_size

        self.iterations = iterations

    def forward(self, x, mask=None):
        b, c, h, w = x.shape  # 1, 64, 24, 24
        q = self.qkv1conv(self.qkv_0(x))  # 2, 64, 24, 24
        k = self.qkv2conv(self.qkv_1(x))  # 2, 64, 24, 24
        v = self.qkv3conv(self.qkv_2(x))  # 2, 64, 24, 24
        if mask is not None:
            mask = get_open_map(mask, self.opr_kernel_size, self.iterations) - mask
            q = q * mask  # 1, 64, 24, 24
            k = k * mask  # 1, 64, 24, 24

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 1, 8, 8, 576
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 1, 8, 8, 576
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 1, 8, 8, 576

        q = torch.nn.functional.normalize(q, dim=-1)  # 1, 8, 8, 576  最后一个维度进行归一化
        k = torch.nn.functional.normalize(k, dim=-1)  # 1, 8, 8, 576
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 1, 8, 8, 8

        # DropKey
        m_r = torch.ones_like(attn) * self.mask_ratio
        attn = attn + torch.bernoulli(m_r) * -1e12

        attn = attn.softmax(dim=-1)
        out = (attn @ v)  # 1, 8, 8, 576
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)  # 1, 64, 24, 24
        out = self.project_out(out)  # 1, 64, 24, 24
        return out

    def initialize(self):
        weight_init(self)


class MSA_head(nn.Module):
    def __init__(self, mode='dilation', dim=128, num_heads=8, ffn_expansion_factor=4, bias=False,
                 LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)


class MSA_module(nn.Module):
    def __init__(self, channel1, channel2):
        super(MSA_module, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2
        dim = channel1
        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.B_TA = MSA_head(dim=dim)
        self.F_TA = MSA_head(dim=dim)
        self.TA = MSA_head(dim=dim)
        self.Fuse = nn.Conv2d(3 * dim, dim, kernel_size=3, padding=1)
        self.Fuse2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

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
        N, C, H, W = x.shape  # C: 64, H: 24, N: 1, W: 24
        mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear')  # 1, 1, 24, 24
        mask_d = mask.detach()  # 1, 1, 24, 24
        mask_d = torch.sigmoid(mask_d)
        xf = self.F_TA(x, mask_d)  # 1,64,24,24
        xb = self.B_TA(x, 1 - mask_d)  # 1,64,24,24
        x = self.TA(x)  # 1,64,24,24
        x = torch.cat((xb, xf, x), 1)  # 1,192,24,24
        x = x.view(N, 3 * C, H, W)  # 1,192,24,24
        x = self.Fuse(x)  # 1,64,24,24
        D = self.Fuse2(side_x + side_x * x)  # 1,64,24,24

        # output_map = self.output_map(D)  # 1, 1, 24, 24

        # return D, output_map
        return D

    def initialize(self):
        weight_init(self)


class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels * 3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels * 2)

        self.conv3 = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False)
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

        self.fuse1 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        self.fuse2 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        self.fuse3 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))

        self.MSA5 = MSA_module(dim=channels)
        self.MSA4 = MSA_module(dim=channels)
        self.MSA3 = MSA_module(dim=channels)
        self.MSA2 = MSA_module(dim=channels)

        self.predtrans1 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans2 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans5 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

        self.initialize()

    def forward(self, E4, E3, E2, E1, shape):
        E4, E3, E2, E1 = self.side_conv1(E4), self.side_conv2(E3), self.side_conv3(E2), self.side_conv4(E1)
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

        E4 = torch.cat((E4, E5), 1)  # 2, 256, 24, 24
        E3 = torch.cat((E3, E5), 1)  # 2, 256, 24, 24
        E2 = torch.cat((E2, E5), 1)  # 2, 256, 24, 24

        E4 = F.relu(self.fuse1(E4), inplace=True)  # 2, 128, 24, 24
        E3 = F.relu(self.fuse2(E3), inplace=True)  # 2, 128, 24, 24
        E2 = F.relu(self.fuse3(E2), inplace=True)  # 2, 128, 24, 24

        P5 = self.predtrans5(E5)  # 2, 1, 24, 24

        D4 = self.MSA5(E5, E4, P5)  # 2, 128, 24, 24
        D4 = F.interpolate(D4, size=E3.size()[2:], mode='bilinear')  # 2, 128, 24, 24
        P4 = self.predtrans4(D4)  # 2, 1, 24, 24

        D3 = self.MSA4(D4, E3, P4)  # 2, 128, 24, 24
        D3 = F.interpolate(D3, size=E2.size()[2:], mode='bilinear')  # 2, 128, 24, 24
        P3 = self.predtrans3(D3)  # 2, 1, 24, 24

        D2 = self.MSA3(D3, E2, P3)  # 2, 128, 24, 24
        D2 = F.interpolate(D2, size=E1.size()[2:], mode='bilinear')  # 2, 128, 96, 96
        P2 = self.predtrans2(D2)  # 2, 1, 96, 96

        D1 = self.MSA2(D2, E1, P2)  # 2, 128, 96, 96
        P1 = self.predtrans1(D1)  # 2, 1, 96, 96

        P1 = F.interpolate(P1, size=shape, mode='bilinear')  # 2, 1, 384, 384
        P2 = F.interpolate(P2, size=shape, mode='bilinear')  # 2, 1, 384, 384
        P3 = F.interpolate(P3, size=shape, mode='bilinear')  # 2, 1, 384, 384
        P4 = F.interpolate(P4, size=shape, mode='bilinear')  # 2, 1, 384, 384
        P5 = F.interpolate(P5, size=shape, mode='bilinear')  # 2, 1, 384, 384

        return P5, P4, P3, P2, P1

    def initialize(self):
        weight_init(self)

