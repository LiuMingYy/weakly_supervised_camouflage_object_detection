import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import sys
from mmcv.cnn import build_norm_layer
sys.path.insert(0, '../../')
from mmcv.ops.carafe import CARAFEPack
from my_exp.model_exp.pvtv2 import pvt_v2_b0,pvt_v2_b1,pvt_v2_b2,pvt_v2_b2_li,pvt_v2_b3,pvt_v2_b4,pvt_v2_b5
# from pvtv2_encoder import pvt_v2_b3
from my_exp.model_exp.decoder_p import MSA_module
# from dyrelu import DyReLUB
from my_exp.model_exp.dyrelu import DyReLUB


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,dilation=1):
        super(CBR, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,dilation=dilation)
        self.norm_cfg = {'type': 'BN', 'requires_grad': True}
        _, self.bn = build_norm_layer(self.norm_cfg, out_channels)
        self.relu = DyReLUB(out_channels, conv_type='2d')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = F.relu(x, inplace=True)
        # x = nn.GELU(x)
        x = self.relu(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=(0,0), dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class OAA(nn.Module):
    def __init__(self,cur_in_channels=64,low_in_channels=32,out_channels=16,cur_scale=2,low_scale=1):
        super(OAA,self).__init__()
        self.relu = DyReLUB(out_channels, conv_type='2d')
        # self.gelu = nn.GELU()
        self.cur_in_channels = cur_in_channels
        self.cur_conv = nn.Sequential(
            nn.Conv2d(in_channels=cur_in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_channels=low_in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )

        self.cur_scale = cur_scale
        self.low_scale = low_scale

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )

        self.branch0 = nn.Sequential(
            BasicConv2d(2 * out_channels, out_channels, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(2 * out_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(2 * out_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(2 * out_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(2 * out_channels, out_channels, 1)

    def forward(self,x_cur,x_low):
        """
        x_cur: 1, 64, 96, 96
        x_low: 1, 128, 48, 48
        """
        x_cur = self.cur_conv(x_cur)  # 1, 32, 96, 96
        # bicubic bilinear nearest
        x_cur = F.interpolate(x_cur, scale_factor=self.cur_scale,  mode='bicubic',align_corners = False)  # 1, 32, 96, 96

        x_low = self.low_conv(x_low)  # 1, 32, 48, 48
        x_low = F.interpolate(x_low, scale_factor=self.low_scale,  mode='bicubic',align_corners = False)  # 1, 32, 96, 96
        x = torch.cat((x_cur,x_low),dim=1)  # 1, 64, 96, 96
        # x = self.out_conv(x)  # 1, 32, 96, 96
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        # x = self.gelu(x_cat + self.conv_res(x))
        x = self.relu(x_cat + self.conv_res(x))

        return x


import numpy as np
import cv2


def get_open_map(input,kernel_size,iterations):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    open_map_list = map(lambda i: cv2.dilate(i.permute(1, 2, 0).detach().numpy(), kernel=kernel, iterations=iterations), input.cpu())
    open_map_tensor = torch.from_numpy(np.array(list(open_map_list)))
    return open_map_tensor.unsqueeze(1).cuda()


class Basic_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Basic_Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FGC(nn.Module):
    def __init__(self, channel1, channel2,focus_background = True, opr_kernel_size = 3,iterations = 1):
        super(FGC, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2
        self.focus_background = focus_background
        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())

        #只用来查看参数
        self.increase_input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=1))
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)
        self.beta = nn.Parameter(torch.ones(1))


        self.conv2 = nn.Conv2d(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3, padding=1,
                               stride=1)

        self.conv_cur_dep1 = Basic_Conv(2 * self.channel1, self.channel1, 3, 1, 1)

        self.conv_cur_dep2 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.conv_cur_dep3 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.opr_kernel_size = opr_kernel_size

        self.iterations = iterations

    def forward(self, cur_x, dep_x, in_map):
        # x; current-level features  cur_x 1, 64, 24, 24
        # y: higher-level features  dep_x 1, 64, 12, 12
        # in_map: higher-level prediction  in_map 1, 1, 12, 12

        dep_x = self.up(dep_x)  # 1, 64, 24, 24

        input_map = self.input_map(in_map)  # 1, 1, 24, 24

        if self.focus_background:
            self.increase_map = self.increase_input_map(get_open_map(input_map, self.opr_kernel_size, self.iterations) - input_map)  # 1, 1, 24, 24
            b_feature = cur_x * self.increase_map  # 当前层中,关注深层部分没有关注的部分  1, 64, 24, 24

        else:
            b_feature = cur_x * input_map  # 在当前层中，对深层部分关注的部分更加关注，同时也关注一下其他部分
        # b_feature = cur_x
        fn = self.conv2(b_feature)  # 1, 64, 24, 24

        refine2 = self.conv_cur_dep1(torch.cat((dep_x, self.beta * fn),dim=1))  # 1, 64, 24, 24
        refine2 = self.conv_cur_dep2(refine2)  # 1, 64, 24, 24
        refine2 = self.conv_cur_dep3(refine2)  # 1, 64, 24, 24

        output_map = self.output_map(refine2)  # 1, 1, 24, 24

        return refine2, output_map


class SARNet(nn.Module):
    def __init__(self, fun_str='pvt_v2_b0'):
        super().__init__()
        self.backbone,embedding_dims = eval(fun_str)()

        self.MSA3 = MSA_module(embedding_dims[3]//8, embedding_dims[3]//8)

        self.MSA2 = MSA_module(embedding_dims[1]//2, embedding_dims[3]//8)

        self.MSA1 = MSA_module(embedding_dims[0]//2, embedding_dims[1]//2)

        self.MSA0 = MSA_module(embedding_dims[0]//4, embedding_dims[0]//2)

        self.oaa0 = OAA(cur_in_channels=embedding_dims[0], low_in_channels=embedding_dims[1],
                        out_channels=embedding_dims[0]//2, cur_scale=1, low_scale=2)
        self.oaa1 = OAA(cur_in_channels=embedding_dims[1], low_in_channels=embedding_dims[2],
                                  out_channels=embedding_dims[1]//2, cur_scale=1, low_scale=2)
        self.oaa2 = OAA(cur_in_channels=embedding_dims[2], low_in_channels=embedding_dims[3],
                                  out_channels=embedding_dims[3] // 8, cur_scale=1, low_scale=2)

        self.cbr = CBR(in_channels=embedding_dims[3], out_channels=embedding_dims[3] // 8,
                                          kernel_size=3, stride=1,
                                          dilation=1, padding=1)

        self.predict_conv = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dims[3] // 8, out_channels=1, kernel_size=3, padding=1, stride=1))

        self.oaa3 = OAA(cur_in_channels=embedding_dims[0]//2, low_in_channels=embedding_dims[3] // 8,
                                             out_channels=embedding_dims[0] // 4, cur_scale=2,
                                             low_scale=16)  # 16

        self.head = nn.ModuleList([
            nn.Conv2d(in_channels=embedding_dims[3] // 8, out_channels=1, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(embedding_dims[3] // 8, 1, 7, 1, 3),
            nn.Conv2d(embedding_dims[1] // 2, 1, 7, 1, 3),
            nn.Conv2d(embedding_dims[0] // 2, 1, 7, 1, 3),
            nn.Conv2d(embedding_dims[0] // 4, 1, 7, 1, 3),
        ])

    def forward(self, x):
        # byxhz
        layer = self.backbone(x)
        """
        0: 1, 64, 96, 96
        1: 1, 128, 48, 48
        2: 1, 320, 24, 24
        3: 1, 512, 12, 12
        """
        s2 = self.oaa0(layer[0], layer[1])  # 1, 32, 96, 96

        s3 = self.oaa1(layer[1], layer[2])  # 1, 64, 48, 48

        s4 = self.oaa2(layer[2], layer[3])  # 1, 64, 24, 24

        s5 = self.cbr(layer[3])  # 1, 64, 12, 12

        s1 = self.oaa3(s2, s5)  # 1, 16, 192, 192

        # predict4 = self.predict_conv(s5)  # 1, 1, 12, 12
        predict4 = self.head[0](s5)

        # focus
        fgc3 = self.MSA3(s5, s4, predict4)  # 1, 64, 24, 24; 1, 1, 24, 24
        predict3 = self.head[1](fgc3)

        fgc2 = self.MSA2(fgc3, s3, predict3)  # 1, 64, 48, 48; 1, 1, 48, 48
        predict2 = self.head[2](fgc2)

        fgc1 = self.MSA1(fgc2, s2, predict2)  # 1, 32, 96, 96; 1, 1, 96, 96
        predict1 = self.head[3](fgc1)

        fgc0 = self.MSA0(fgc1, s1, predict1)  # 1, 16, 192, 192; 1, 1, 192, 192
        predict0 = self.head[4](fgc0)

        # rescale

        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)  # 1, 1, 384, 384
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)  # 1, 1, 384, 384
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)  # 1, 1, 384, 384
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)  # 1, 1, 384, 384
        predict0 = F.interpolate(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)  # 1, 1, 384, 384

        return predict0, None, predict1, predict2, predict3, predict4


if __name__ =='__main__':
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        torch.cuda.init()

    from thop import profile
    net = SARNet('pvt_v2_b3').cuda()
    data = torch.randn(1, 3, 320, 320).cuda()
    flops, params = profile(net, (data,))
    print('flops: %.2f G, params: %.2f M' % (flops / (1024*1024*1024), params / (1024*1024)))
    y = net(data)
    for i in y:
        print(i.shape)


