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
# from dyrelu import DyReLUB
from my_exp.model_exp.dyrelu import DyReLUB


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


class Attention(nn.Module):
    def __init__(self, dim, mode, num_heads, bias, opr_kernel_size=3,iterations=1):
        super(Attention, self).__init__()
        self.mode = mode
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.mask_ratio = torch.tensor(0.3)

        self.opr_kernel_size = opr_kernel_size

        self.iterations = iterations

    def forward(self, x, mask=None):
        b,c,h,w = x.shape  # 1, 64, 24, 24

        q=self.qkv1conv(self.qkv_0(x))  # 1, 64, 24, 24
        k=self.qkv2conv(self.qkv_1(x))  # 1, 64, 24, 24
        v=self.qkv3conv(self.qkv_2(x))  # 1, 64, 24, 24

        if mask is not None:
            mask = get_open_map(mask, self.opr_kernel_size, self.iterations) - mask  # 只关注背景边缘
            q=q*mask  # 1, 64, 24, 24
            k=k*mask  # 1, 64, 24, 24

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
    def __init__(self, dim=128, mode='', num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.attn = Attention(dim, mode, num_heads, bias)
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
        # self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
        #                         nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), DyReLUB(self.channel1, conv_type='2d'), nn.UpsamplingBilinear2d(scale_factor=2))
        # self.Sep = nn.Conv2d(dim, 3*dim, kernel_size=3, padding=1)
        self.B_TA = MSA_head(dim=dim,mode='b')
        self.F_TA = MSA_head(dim=dim,mode='f')
        self.TA = MSA_head(dim=dim,mode='t')
        self.Fuse = nn.Conv2d(3*dim, dim, kernel_size=3, padding=1)
        # self.Fuse2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.Fuse2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1), nn.BatchNorm2d(dim), DyReLUB(dim, conv_type='2d'))

    def forward(self, x, side_x, mask):
        """

        Args:
            x: 1, 64, 12, 12
            side_x: 1, 64, 24, 24
            mask: 1, 1, 12, 12

        Returns:

        """

        x = self.up(x)  # 1,64,24,24
        N,C,H,W = x.shape  # C: 64, H: 24, N: 1, W: 24
        # xf, xb, xt = self.Sep(x).chunk(3, dim=1)
        mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear')  # 1, 1, 24, 24
        mask_d = mask.detach()  # 1, 1, 24, 24
        mask_d = torch.sigmoid(mask_d)

        xf = self.F_TA(x, mask_d)  # 1,64,24,24
        xb = self.B_TA(x, 1-mask_d)  # 1,64,24,24
        xt = self.TA(x)  # 1,64,24,24
        x = torch.cat((xb, xf, xt),1)  # 1,192,24,24
        x = x.view(N,3*C,H,W)  # 1,192,24,24
        x = self.Fuse(x)  # 1,64,24,24
        D = self.Fuse2(side_x+side_x*x)  # 1,64,24,24

        return D
    
    def initialize(self):
        weight_init(self)


 
