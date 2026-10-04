#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Jose Dolz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def random_weights_init_3d(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def conv_block_3d(in_dim, out_dim, **kwconv):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, **kwconv),
        nn.BatchNorm3d(out_dim),
        nn.PReLU(),
    )


def conv_block_asym_3d(in_dim, out_dim, *, kernel_size: int):
    """3D Asymmetric convolution decomposed across spatial dimensions (Z, H, W)."""
    p = kernel_size // 2
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=(kernel_size, 1, 1), padding=(p, 0, 0)),
        nn.Conv3d(out_dim, out_dim, kernel_size=(1, kernel_size, 1), padding=(0, p, 0)),
        nn.Conv3d(out_dim, out_dim, kernel_size=(1, 1, kernel_size), padding=(0, 0, p)),
        nn.BatchNorm3d(out_dim),
        nn.PReLU(),
    )


class BottleNeck3D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        projectionFactor: int,
        *,
        dropoutRate: float = 0.01,
        dilation: int = 1,
        asym: bool = False,
        dilate_last: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        mid_dim: int = in_dim // projectionFactor

        # Secondary branch
        self.block0 = conv_block_3d(in_dim, mid_dim, kernel_size=1)

        if not asym:
            self.block1 = conv_block_3d(
                mid_dim, mid_dim, kernel_size=3, padding=dilation, dilation=dilation
            )
        else:
            self.block1 = conv_block_asym_3d(mid_dim, mid_dim, kernel_size=5)

        self.block2 = conv_block_3d(mid_dim, out_dim, kernel_size=1)

        self.do = nn.Dropout3d(p=dropoutRate)
        self.PReLU_out = nn.PReLU()

        if in_dim > out_dim:
            self.conv_out = conv_block_3d(in_dim, out_dim, kernel_size=1)
        elif dilate_last:
            self.conv_out = conv_block_3d(in_dim, out_dim, kernel_size=3, padding=1)
        else:
            self.conv_out = nn.Identity()

    def forward(self, in_: Tensor) -> Tensor:
        b0 = self.block0(in_)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        output = self.PReLU_out(self.conv_out(in_) + do)
        return output


class BottleNeckDownSampling3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, projectionFactor: int):
        super().__init__()
        mid_dim: int = in_dim // projectionFactor

        # Main branch
        self.maxpool0 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)

        # Secondary branch
        self.block0 = conv_block_3d(in_dim, mid_dim, kernel_size=2, padding=0, stride=2)
        self.block1 = conv_block_3d(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.block2 = conv_block_3d(mid_dim, out_dim, kernel_size=1)

        # Regularizer
        self.do = nn.Dropout3d(p=0.01)
        self.PReLU = nn.PReLU()

    def forward(self, in_: Tensor) -> tuple[Tensor, Tensor]:
        # Main branch
        maxpool_output, indices = self.maxpool0(in_)

        # Secondary branch
        b0 = self.block0(in_)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        _, c, _, _, _ = maxpool_output.shape
        output = do
        output[:, :c, :, :, :] += maxpool_output

        final_output = self.PReLU(output)
        return final_output, indices


class BottleNeckUpSampling3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, projectionFactor: int):
        super().__init__()
        mid_dim: int = in_dim // projectionFactor

        # Main branch
        self.unpool = nn.MaxUnpool3d(kernel_size=2)

        # Secondary branch
        self.block0 = conv_block_3d(in_dim, mid_dim, kernel_size=3, padding=1)
        self.block1 = conv_block_3d(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.block2 = conv_block_3d(mid_dim, out_dim, kernel_size=1)

        # Regularizer
        self.do = nn.Dropout3d(p=0.01)
        self.PReLU = nn.PReLU()

    def forward(self, args: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        in_, indices, skip = args

        # Main branch: pass skip's output_size to ensure exact tensor dimension matching
        up = self.unpool(in_, indices, output_size=skip.size())

        # Secondary branch
        b0 = self.block0(torch.cat((up, skip), dim=1))
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        output = self.PReLU(up + do)
        return output


class ChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, input: Tensor) -> Tensor:
        batch, channels, _, _, _ = input.shape
        weights = self.gate(self.pool(input).view(batch, channels))
        return input * weights.view(batch, channels, 1, 1, 1)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.gate = nn.Conv3d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.activation = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        average = input.mean(dim=1, keepdim=True)
        maximum = input.amax(dim=1, keepdim=True)
        weights = self.activation(self.gate(torch.cat((average, maximum), dim=1)))
        return input * weights


class ENet3D(nn.Module):
    """Full 3D ENet Architecture for volumetric segmentation [B, C, Z, H, W]."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        F_factor: int = kwargs.get("factor", 4)
        K: int = kwargs.get("kernels", 16)

        # Initial operations - use kernel_size=3, stride=2, padding=1 for both to guarantee matching spatial output
        self.conv0 = nn.Conv3d(in_dim, K - 1, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, return_indices=False)

        # Downsampling encoder
        self.bottleneck1_0 = BottleNeckDownSampling3D(K, K * 4, F_factor)
        self.bottleneck1_1 = nn.Sequential(
            BottleNeck3D(K * 4, K * 4, F_factor),
            BottleNeck3D(K * 4, K * 4, F_factor),
            BottleNeck3D(K * 4, K * 4, F_factor),
            BottleNeck3D(K * 4, K * 4, F_factor),
        )
        self.bottleneck2_0 = BottleNeckDownSampling3D(K * 4, K * 8, F_factor)
        self.bottleneck2_1 = nn.Sequential(
            BottleNeck3D(K * 8, K * 8, F_factor, dropoutRate=0.1),
            BottleNeck3D(K * 8, K * 8, F_factor, dilation=2),
            BottleNeck3D(K * 8, K * 8, F_factor, dropoutRate=0.1, asym=True),
            BottleNeck3D(K * 8, K * 8, F_factor, dilation=4),
            BottleNeck3D(K * 8, K * 8, F_factor, dropoutRate=0.1),
            BottleNeck3D(K * 8, K * 8, F_factor, dilation=8),
            BottleNeck3D(K * 8, K * 8, F_factor, dropoutRate=0.1, asym=True),
            BottleNeck3D(K * 8, K * 8, F_factor, dilation=16),
        )

        # Middle bottleneck
        self.bottleneck3 = nn.Sequential(
            BottleNeck3D(K * 8, K * 8, F_factor, dropoutRate=0.1),
            BottleNeck3D(K * 8, K * 8, F_factor, dilation=2),
            BottleNeck3D(K * 8, K * 8, F_factor, dropoutRate=0.1, asym=True),
            BottleNeck3D(K * 8, K * 8, F_factor, dilation=4),
            BottleNeck3D(K * 8, K * 8, F_factor, dropoutRate=0.1),
            BottleNeck3D(K * 8, K * 8, F_factor, dilation=8),
            BottleNeck3D(K * 8, K * 8, F_factor, dropoutRate=0.1, asym=True),
            BottleNeck3D(K * 8, K * 4, F_factor, dilation=16, dilate_last=True),
        )

        # Upsampling decoder
        self.bottleneck4 = nn.Sequential(
            BottleNeckUpSampling3D(K * 8, K * 4, F_factor),
            BottleNeck3D(K * 4, K * 4, F_factor, dropoutRate=0.1),
            BottleNeck3D(K * 4, K, F_factor, dropoutRate=0.1),
        )
        self.bottleneck5 = nn.Sequential(
            BottleNeckUpSampling3D(K * 2, K, F_factor),
            BottleNeck3D(K, K, F_factor, dropoutRate=0.1),
        )

        # Final convolutions
        self.final = nn.Sequential(
            conv_block_3d(K, K, kernel_size=3, padding=1, bias=False, stride=1),
            conv_block_3d(K, K, kernel_size=3, padding=1, bias=False, stride=1),
            nn.Conv3d(K, out_dim, kernel_size=1),
        )

        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

    def forward(self, input: Tensor) -> Tensor:
        # Initial operations
        conv_0 = self.conv0(input)
        maxpool_0 = self.maxpool0(input)
        output_initial = torch.cat((conv_0, maxpool_0), dim=1)

        # Downsampling half
        bn1_0, indices_1 = self.bottleneck1_0(output_initial)
        bn1_out = self.bottleneck1_1(bn1_0)
        bn2_0, indices_2 = self.bottleneck2_0(bn1_out)
        bn2_out = self.bottleneck2_1(bn2_0)

        # Middle operations
        bn3_out = self.bottleneck3(bn2_out)

        # Upsampling half
        bn4_out = self.bottleneck4((bn3_out, indices_2, bn1_out))
        bn5_out = self.bottleneck5((bn4_out, indices_1, output_initial))

        # Final upsampling matched explicitly to original input shape
        interpolated = F.interpolate(
            bn5_out, size=input.shape[2:], mode="trilinear", align_corners=False
        )
        return self.final(interpolated)

    def init_weights(self, *args, **kwargs):
        self.apply(random_weights_init_3d)


class AttentionENet3D(ENet3D):
    """3D ENet with channel attention at the bottleneck feature scale."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        kernels = kwargs.get("kernels", 16)
        self.attention_down = ChannelAttention3D(kernels * 8)
        self.attention_middle = ChannelAttention3D(kernels * 4)

    def forward(self, input: Tensor) -> Tensor:
        conv_0 = self.conv0(input)
        maxpool_0 = self.maxpool0(input)
        output_initial = torch.cat((conv_0, maxpool_0), dim=1)

        bn1_0, indices_1 = self.bottleneck1_0(output_initial)
        bn1_out = self.bottleneck1_1(bn1_0)
        bn2_0, indices_2 = self.bottleneck2_0(bn1_out)
        bn2_out = self.bottleneck2_1(bn2_0)

        bn2_out = self.attention_down(bn2_out)
        bn3_out = self.bottleneck3(bn2_out)
        bn3_out = self.attention_middle(bn3_out)

        bn4_out = self.bottleneck4((bn3_out, indices_2, bn1_out))
        bn5_out = self.bottleneck5((bn4_out, indices_1, output_initial))

        interpolated = F.interpolate(
            bn5_out, size=input.shape[2:], mode="trilinear", align_corners=False
        )
        return self.final(interpolated)