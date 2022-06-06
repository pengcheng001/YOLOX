#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(3, 3, 3), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m3 = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks//2)
                for ks in kernel_sizes
            ]
        )
        kernel_sizes = list(kernel_sizes)[:2]
        self.m2 = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        kernel_sizes = list(kernel_sizes)[:1]
        self.m1 = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        kernel_sizes = (3, 3, 3)
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)
        # self.zeropad = nn.ZeroPad2d(padding=(1, 0, 1, 0))

    def forward(self, x):
        x = self.conv1(x)
        #data = self.zeropad(x)
        x_m3 = self.m3[0](self.m3[1](self.m3[2](x)))
        x_m2 = self.m2[0](self.m2[1](x))
        x_m1 = self.m1[0](x)

        x = torch.cat([x]+[x_m1, x_m2, x_m3], dim=1)
        #x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


# class SPPBottleneck(nn.Module):
#     """Spatial pyramid pooling layer used in YOLOv3-SPP"""

#     def __init__(
#         self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
#     ):
#         super().__init__()
#         hidden_channels = in_channels // 2
#         self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
#         self.m = nn.ModuleList(
#             [
#                 nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
#                 for ks in kernel_sizes
#             ]
#         )
#         conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
#         self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = torch.cat([x] + [m(x) for m in self.m], dim=1)
#         x = self.conv2(x)
#         return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


# class Focus(nn.Module):
#     """Focus width and height information into channel space."""

#     def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
#         super().__init__()
#         self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

#     def forward(self, x):
#         # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
#         # patch_top_left = x[..., ::2, ::2]
#         # patch_top_right = x[..., ::2, 1::2]
#         # patch_bot_left = x[..., 1::2, ::2]
#         # patch_bot_right = x[..., 1::2, 1::2]
#         # x = torch.cat(
#         #     (
#         #         patch_top_left,
#         #         patch_bot_left,
#         #         patch_top_right,
#         #         patch_bot_right,
#         #     ),
#         #     dim=1,
#         # )
#         a, b = x[..., ::2, :].transpose(-2, -1), x[..., 1::2, :].transpose(-2, -1)
#         x = torch.cat([a[..., ::2, :], b[..., ::2, :], a[..., 1::2, :],
#                       b[..., 1::2, :]], 1).transpose(-2, -1)
#         return self.conv(x)

class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)
        #####################################################################################################
        kernel_top_top = torch.zeros((3, 1, 1, 1))
        kernel_top_top[..., 0, 0] = 1
        kernel_top_top = nn.Parameter(kernel_top_top)

        kernel_top_left = torch.zeros((3, 1, 2, 2))
        kernel_top_left[..., 0, 0] = 1
        kernel_top_left = nn.Parameter(kernel_top_left)

        kernel_top_right = torch.zeros((3, 1, 2, 2))
        kernel_top_right[..., 0, 1] = 1
        kernel_top_right = nn.Parameter(kernel_top_right)

        kernel_bot_left = torch.zeros((3, 1, 2, 2))
        kernel_bot_left[..., 1, 0] = 1
        kernel_bot_left = nn.Parameter(kernel_bot_left)

        kernel_bot_right = torch.zeros((3, 1, 2, 2))
        kernel_bot_right[..., 1, 1] = 1
        kernel_bot_right = nn.Parameter(kernel_bot_right)

        self.top_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(
            1, 1), stride=(1, 1), groups=3, bias=False)
        self.top_conv.weight = kernel_top_top

        self.conv_top_left_lx = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2, 2), stride=(2, 2), groups=3,
                                          bias=False)
        self.conv_top_left_lx.weight = kernel_top_left

        self.conv_top_right_lx = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2, 2), stride=(2, 2), groups=3,
                                           bias=False)
        self.conv_top_right_lx.weight = kernel_top_right

        self.conv_bot_left_lx = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2, 2), stride=(2, 2), groups=3,
                                          bias=False)
        self.conv_bot_left_lx.weight = kernel_bot_left

        self.conv_bot_right_lx = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2, 2), stride=(2, 2), groups=3,
                                           bias=False)
        self.conv_bot_right_lx.weight = kernel_bot_right

        for k, m in self.named_parameters():
            if ("top" in k or "bot" in k) and ('right' in k or 'left' in k) or ("top_conv" in k):
                m.requires_grad = False

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        # patch_top_left = x[..., ::2, ::2]
        # patch_top_right = x[..., ::2, 1::2]
        # patch_bot_left = x[..., 1::2, ::2]
        # patch_bot_right = x[..., 1::2, 1::2]
        # x = torch.cat(
        #     (
        #         patch_top_left,
        #         patch_bot_left,
        #         patch_top_right,
        #         patch_bot_right,
        #     ),
        #     dim=1,
        # )
        # a, b = x[..., ::2, :].transpose(-2, -1), x[..., 1::2, :].transpose(-2, -1)
        # x = torch.cat([a[..., ::2, :], b[..., ::2, :], a[..., 1::2, :],
        #               b[..., 1::2, :]], 1).transpose(-2, -1)
        # for tda4
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        x = self.top_conv(x)
        patch_top_left = self.conv_top_left_lx(x)
        patch_top_right = self.conv_top_right_lx(x)
        patch_bot_left = self.conv_bot_left_lx(x)
        patch_bot_right = self.conv_bot_right_lx(x)
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)
