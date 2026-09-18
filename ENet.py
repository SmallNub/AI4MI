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


def random_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def conv_block(in_dim, out_dim, **kwconv):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, **kwconv), nn.BatchNorm2d(out_dim), nn.PReLU()
    )


def conv_block_asym(in_dim, out_dim, *, kernel_size: int):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size, 1), padding=(2, 0)),
        nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size), padding=(0, 2)),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )


class BottleNeck(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        projectionFactor,
        *,
        dropoutRate=0.01,
        dilation=1,
        asym: bool = False,
        dilate_last: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        mid_dim: int = in_dim // projectionFactor

        # Main branch

        # Secondary branch
        self.block0 = conv_block(in_dim, mid_dim, kernel_size=1)

        if not asym:
            self.block1 = conv_block(
                mid_dim, mid_dim, kernel_size=3, padding=dilation, dilation=dilation
            )
        else:
            self.block1 = conv_block_asym(mid_dim, mid_dim, kernel_size=5)

        self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()

        if in_dim > out_dim:
            self.conv_out = conv_block(in_dim, out_dim, kernel_size=1)
        elif dilate_last:
            self.conv_out = conv_block(in_dim, out_dim, kernel_size=3, padding=1)
        else:
            self.conv_out = nn.Identity()

    def forward(self, in_) -> Tensor:
        # Main branch
        # Secondary branch
        b0 = self.block0(in_)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        output = self.PReLU_out(self.conv_out(in_) + do)

        return output


class BottleNeckDownSampling(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor):
        super().__init__()
        mid_dim: int = in_dim // projectionFactor

        # Main branch
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)

        # Secondary branch
        self.block0 = conv_block(in_dim, mid_dim, kernel_size=2, padding=0, stride=2)
        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

        # Regularizer
        self.do = nn.Dropout(p=0.01)
        self.PReLU = nn.PReLU()

        # Out

    def forward(self, in_) -> tuple[Tensor, Tensor]:
        # Main branch
        maxpool_output, indices = self.maxpool0(in_)

        # Secondary branch
        b0 = self.block0(in_)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        _, c, _, _ = maxpool_output.shape
        output = do
        output[:, :c, :, :] += maxpool_output

        final_output = self.PReLU(output)

        return final_output, indices


class BottleNeckUpSampling(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor):
        super().__init__()
        mid_dim: int = in_dim // projectionFactor

        # Main branch
        self.unpool = nn.MaxUnpool2d(2)

        # Secondary branch
        self.block0 = conv_block(in_dim, mid_dim, kernel_size=3, padding=1)
        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

        # Regularizer
        self.do = nn.Dropout(p=0.01)
        self.PReLU = nn.PReLU()

        # Out

    def forward(self, args) -> Tensor:
        # nn.Sequential cannot handle multiple parameters:
        in_, indices, skip = args

        # Main branch
        up = self.unpool(in_, indices)

        # Secondary branch
        b0 = self.block0(torch.cat((up, skip), dim=1))
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        output = self.PReLU(up + do)

        return output


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, input: Tensor) -> Tensor:
        batch, channels, _, _ = input.shape
        weights = self.gate(self.pool(input).view(batch, channels))
        return input * weights.view(batch, channels, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.gate = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.activation = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        average = input.mean(dim=1, keepdim=True)
        maximum = input.amax(dim=1, keepdim=True)
        weights = self.activation(self.gate(torch.cat((average, maximum), dim=1)))
        return input * weights


def _enet_features(
    net: "ENet", input: Tensor
) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]:
    conv_0 = net.conv0(input)
    maxpool_0 = net.maxpool0(input)
    output_initial = torch.cat((conv_0, maxpool_0), dim=1)

    bn1_0, indices_1 = net.bottleneck1_0(output_initial)
    bn1_out = net.bottleneck1_1(bn1_0)
    bn2_0, indices_2 = net.bottleneck2_0(bn1_out)
    bn2_out = net.bottleneck2_1(bn2_0)

    return output_initial, bn1_out, bn2_out, (indices_1, indices_2)


def _enet_decode(
    net: "ENet",
    output_initial: Tensor,
    bn1_out: Tensor,
    bn3_out: Tensor,
    indices: tuple[Tensor, Tensor],
) -> Tensor:
    indices_1, indices_2 = indices
    bn4_out = net.bottleneck4((bn3_out, indices_2, bn1_out))
    bn5_out = net.bottleneck5((bn4_out, indices_1, output_initial))
    interpolated = F.interpolate(bn5_out, mode="nearest", scale_factor=2)
    return net.final(interpolated)


class ENet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        F: int = kwargs["factor"] if "factor" in kwargs else 4  # Projecting factor
        K: int = kwargs["kernels"] if "kernels" in kwargs else 16  # n_kernels

        # from models.enet import (BottleNeck,
        #                          BottleNeckDownSampling,
        #                          BottleNeckUpSampling,
        #                          conv_block)

        # Initial operations
        self.conv0 = nn.Conv2d(in_dim, K - 1, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool2d(2, return_indices=False, ceil_mode=False)

        # Downsampling half
        self.bottleneck1_0 = BottleNeckDownSampling(K, K * 4, F)
        self.bottleneck1_1 = nn.Sequential(
            BottleNeck(K * 4, K * 4, F),
            BottleNeck(K * 4, K * 4, F),
            BottleNeck(K * 4, K * 4, F),
            BottleNeck(K * 4, K * 4, F),
        )
        self.bottleneck2_0 = BottleNeckDownSampling(K * 4, K * 8, F)
        self.bottleneck2_1 = nn.Sequential(
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
            BottleNeck(K * 8, K * 8, F, dilation=2),
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
            BottleNeck(K * 8, K * 8, F, dilation=4),
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
            BottleNeck(K * 8, K * 8, F, dilation=8),
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
            BottleNeck(K * 8, K * 8, F, dilation=16),
        )

        # Middle operations
        self.bottleneck3 = nn.Sequential(
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
            BottleNeck(K * 8, K * 8, F, dilation=2),
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
            BottleNeck(K * 8, K * 8, F, dilation=4),
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
            BottleNeck(K * 8, K * 8, F, dilation=8),
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
            BottleNeck(K * 8, K * 4, F, dilation=16, dilate_last=True),
        )

        # Upsampling half
        self.bottleneck4 = nn.Sequential(
            BottleNeckUpSampling(K * 8, K * 4, F),
            BottleNeck(K * 4, K * 4, F, dropoutRate=0.1),
            BottleNeck(K * 4, K, F, dropoutRate=0.1),
        )
        self.bottleneck5 = nn.Sequential(
            BottleNeckUpSampling(K * 2, K, F), BottleNeck(K, K, F, dropoutRate=0.1)
        )

        # Final upsampling and covolutions
        self.final = nn.Sequential(
            conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
            conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
            nn.Conv2d(K, out_dim, kernel_size=1),
        )

        print(
            f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}"
        )

    def forward(self, input):
        # Initial operations
        conv_0 = self.conv0(input)
        maxpool_0 = self.maxpool0(input)
        outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

        # Downsampling half
        bn1_0, indices_1 = self.bottleneck1_0(outputInitial)
        bn1_out = self.bottleneck1_1(bn1_0)
        bn2_0, indices_2 = self.bottleneck2_0(bn1_out)
        bn2_out = self.bottleneck2_1(bn2_0)

        # Middle operations
        bn3_out = self.bottleneck3(bn2_out)

        # Upsampling half
        bn4_out = self.bottleneck4((bn3_out, indices_2, bn1_out))
        bn5_out = self.bottleneck5((bn4_out, indices_1, outputInitial))

        # Final upsampling and covolutions
        interpolated = F.interpolate(bn5_out, mode="nearest", scale_factor=2)
        return self.final(interpolated)

    def init_weights(self, *args, **kwargs):
        self.apply(random_weights_init)


class AttentionENet(ENet):
    """ENet with channel attention at the bottleneck feature scale."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        kernels = kwargs["kernels"] if "kernels" in kwargs else 16
        self.attention_down = ChannelAttention(kernels * 8)
        self.attention_middle = ChannelAttention(kernels * 4)

    def forward(self, input: Tensor) -> Tensor:
        output_initial, bn1_out, bn2_out, indices = _enet_features(self, input)
        bn2_out = self.attention_down(bn2_out)
        bn3_out = self.bottleneck3(bn2_out)
        bn3_out = self.attention_middle(bn3_out)
        return _enet_decode(
            self,
            output_initial,
            bn1_out,
            bn3_out,
            indices,
        )

    def init_weights(self, *args, **kwargs):
        super().init_weights(*args, **kwargs)
        for module in (self.attention_down, self.attention_middle):
            for layer in module.gate:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)


class SpatialENet(ENet):
    """ENet with spatial attention at the bottleneck feature scale."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        kernels = kwargs["kernels"] if "kernels" in kwargs else 16
        self.attention_down = SpatialAttention()
        self.attention_middle = SpatialAttention()
        self.channels_down = kernels * 8
        self.channels_middle = kernels * 4

    def forward(self, input: Tensor) -> Tensor:
        output_initial, bn1_out, bn2_out, indices = _enet_features(self, input)
        bn2_out = self.attention_down(bn2_out)
        bn3_out = self.attention_middle(self.bottleneck3(bn2_out))
        return _enet_decode(
            self,
            output_initial,
            bn1_out,
            bn3_out,
            indices,
        )


class CBAMENet(AttentionENet):
    """ENet with channel and spatial attention at the bottleneck scale."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        self.spatial_down = SpatialAttention()
        self.spatial_middle = SpatialAttention()

    def forward(self, input: Tensor) -> Tensor:
        output_initial, bn1_out, bn2_out, indices = _enet_features(self, input)
        refined = self.attention_down(bn2_out)
        refined = self.spatial_down(refined)
        refined = self.bottleneck3(refined)
        refined = self.attention_middle(refined)
        refined = self.spatial_middle(refined)
        return _enet_decode(self, output_initial, bn1_out, refined, indices)


class FuseBlock(nn.Module):
    def __init__(self, channels: int, z_window: int):
        super().__init__()
        self.fuse = nn.Conv3d(
            channels, channels, kernel_size=(z_window, 1, 1), bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    [B, C, Z, H, W]
        x = self.fuse(x)  #       [B, C, 1, H, W]
        x = x.squeeze(2)  #       [B, C, H, W]
        return self.act(self.bn(x))


class LateFusionENet(ENet):
    def __init__(self, in_dim: int, out_dim: int, z_window: int = 3, **kwargs):
        super().__init__(in_dim=1, out_dim=out_dim, **kwargs)

        self.z_window = z_window
        self.center_idx = z_window // 2
        K = kwargs.get("kernels", 16)

        self.fusion = FuseBlock(channels=K * 4, z_window=z_window)

    def _extract_center(self, tensor: torch.Tensor, B: int) -> torch.Tensor:
        # Only grab the center slice from Z dims
        _, C, H, W = tensor.shape
        unflattened = tensor.view(B, self.z_window, C, H, W)
        return unflattened[:, self.center_idx, ...]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, Z, C_in, H, W = input.shape
        assert Z == self.z_window, f"Expected Z={self.z_window}, got {Z}"

        x_flat = input.view(B * Z, C_in, H, W)
        output_initial, bn1_out, bn2_out, indices = _enet_features(self, x_flat)
        bn3_out = self.bottleneck3(bn2_out)

        _, C_feat, H_feat, W_feat = bn3_out.shape
        bn3_seq = bn3_out.view(B, Z, C_feat, H_feat, W_feat).permute(0, 2, 1, 3, 4)
        fused_bn3 = self.fusion(bn3_seq)

        center_output_initial = self._extract_center(output_initial, B)
        center_bn1_out = self._extract_center(bn1_out, B)

        center_indices = (
            self._extract_center(indices[0], B),
            self._extract_center(indices[1], B),
        )

        return _enet_decode(
            self, center_output_initial, center_bn1_out, fused_bn3, center_indices
        )
