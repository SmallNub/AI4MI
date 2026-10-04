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


# =====================================================================
# Weight Initialization & Basic Blocks
# =====================================================================

def init_weights_3d(m: nn.Module) -> None:
    """Kaiming/He weight initialization optimized for PReLU activations."""
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def conv_block_3d(in_dim: int, out_dim: int, **kwconv) -> nn.Sequential:
    """Standard 3D Convolution block with InstanceNorm3d and PReLU."""
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, **kwconv),
        nn.InstanceNorm3d(out_dim, affine=True),
        nn.PReLU(),
    )


def conv_block_asym_3d(in_dim: int, out_dim: int, *, kernel_size: int) -> nn.Sequential:
    """3D Asymmetric convolution decomposed across spatial dimensions (Z, H, W)."""
    p = kernel_size // 2
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=(kernel_size, 1, 1), padding=(p, 0, 0)),
        nn.Conv3d(out_dim, out_dim, kernel_size=(1, kernel_size, 1), padding=(0, p, 0)),
        nn.Conv3d(out_dim, out_dim, kernel_size=(1, 1, kernel_size), padding=(0, 0, p)),
        nn.InstanceNorm3d(out_dim, affine=True),
        nn.PReLU(),
    )


# =====================================================================
# Attention & Channel Calibration Modules
# =====================================================================

class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation channel attention module."""
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class AttentionGate3D(nn.Module):
    """3D Attention Gate filtering skip features with deeper gating signals."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1, affine=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        # Align spatial dimensions if upsampling factor mismatch occurs
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="trilinear", align_corners=False)

        psi = self.relu(g1 + x1)
        weight = self.psi(psi)
        return x * weight


# =====================================================================
# Bottlenecks
# =====================================================================

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
        use_se: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        mid_dim: int = max(1, in_dim // projectionFactor)

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
        self.se = SEBlock3D(out_dim, reduction=16) if use_se else nn.Identity()
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
        se = self.se(do)

        output = self.PReLU_out(self.conv_out(in_) + se)
        return output


class BottleNeckDownSampling3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, projectionFactor: int):
        super().__init__()
        mid_dim: int = max(1, in_dim // projectionFactor)

        # Main branch
        self.maxpool0 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=False)

        # Secondary branch
        self.block0 = conv_block_3d(in_dim, mid_dim, kernel_size=2, padding=0, stride=2)
        self.block1 = conv_block_3d(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.block2 = conv_block_3d(mid_dim, out_dim, kernel_size=1)

        # Regularizers & SE
        self.do = nn.Dropout3d(p=0.01)
        self.se = SEBlock3D(out_dim, reduction=16)
        self.PReLU = nn.PReLU()

    def forward(self, in_: Tensor) -> Tensor:
        maxpool_output = self.maxpool0(in_)

        b0 = self.block0(in_)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)
        se = self.se(do)

        _, c, _, _, _ = maxpool_output.shape
        output = se
        output[:, :c, :, :, :] += maxpool_output

        return self.PReLU(output)


class AttentionSkipUpSampling3D(nn.Module):
    """Replaces index unpooling with trilinear upsampling and attention-gated feature concatenation."""
    def __init__(self, in_dim: int, skip_dim: int, out_dim: int, projectionFactor: int):
        super().__init__()
        mid_dim: int = max(1, (in_dim + skip_dim) // projectionFactor)

        # Attention Gate filtering skip features
        self.attn = AttentionGate3D(F_g=in_dim, F_l=skip_dim, F_int=max(1, skip_dim // 2))

        # Secondary branch processing concatenated features
        self.block0 = conv_block_3d(in_dim + skip_dim, mid_dim, kernel_size=3, padding=1)
        self.block1 = conv_block_3d(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.block2 = conv_block_3d(mid_dim, out_dim, kernel_size=1)

        # Identity shortcut projection if dimensions differ
        self.shortcut_conv = (
            conv_block_3d(in_dim, out_dim, kernel_size=1)
            if in_dim != out_dim
            else nn.Identity()
        )

        self.do = nn.Dropout3d(p=0.01)
        self.PReLU = nn.PReLU()

    def forward(self, in_: Tensor, skip: Tensor) -> Tensor:
        # Upsample deep decoder features to match skip spatial scale
        up = F.interpolate(in_, size=skip.shape[2:], mode="trilinear", align_corners=False)

        # Apply Attention Gate to high-resolution encoder skip
        gated_skip = self.attn(g=in_, x=skip)

        # Concatenate upsampled features and gated skip
        combined = torch.cat((up, gated_skip), dim=1)

        b0 = self.block0(combined)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        # Residual update
        shortcut = self.shortcut_conv(up)
        return self.PReLU(shortcut + do)


# =====================================================================
# Full Architecture
# =====================================================================

class ENet3D(nn.Module):
    """Full 3D ENet Architecture with SE-Bottlenecks, Attention-Gated Skips, and Deep Supervision."""

    def __init__(self, in_dim: int = 1, out_dim: int = 5, **kwargs):
        super().__init__()
        F_factor: int = kwargs.get("factor", 4)
        K: int = kwargs.get("kernels", 32)

        # Initial stem
        self.conv0 = nn.Conv3d(in_dim, K - 1, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

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
            BottleNeck3D(K * 8, K * 8, F_factor, dilation=16, dilate_last=True),
        )

        # Attention-Gated Decoder
        self.bottleneck4 = AttentionSkipUpSampling3D(
            in_dim=K * 8, skip_dim=K * 4, out_dim=K * 4, projectionFactor=F_factor
        )
        self.bottleneck4_refine = nn.Sequential(
            BottleNeck3D(K * 4, K * 4, F_factor, dropoutRate=0.1),
            BottleNeck3D(K * 4, K * 2, F_factor, dropoutRate=0.1),
        )

        self.bottleneck5 = AttentionSkipUpSampling3D(
            in_dim=K * 2, skip_dim=K, out_dim=K, projectionFactor=F_factor
        )
        self.bottleneck5_refine = BottleNeck3D(K, K, F_factor, dropoutRate=0.1)

        # Deep Supervision auxiliary head attached at bottleneck4 output
        self.aux_classifier = nn.Sequential(
            conv_block_3d(K * 2, K, kernel_size=3, padding=1),
            nn.Conv3d(K, out_dim, kernel_size=1),
        )

        # Main decoder pre-classifier (computed at lower spatial scale)
        self.pre_final = nn.Sequential(
            conv_block_3d(K, K, kernel_size=3, padding=1, bias=False),
            conv_block_3d(K, K, kernel_size=3, padding=1, bias=False),
        )
        self.classifier = nn.Conv3d(K, out_dim, kernel_size=1)

        self.init_weights()
        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        # Initial operations
        conv_0 = self.conv0(input)
        maxpool_0 = self.maxpool0(input)
        output_initial = torch.cat((conv_0, maxpool_0), dim=1)  # K channels

        # Encoder downsampling
        bn1_0 = self.bottleneck1_0(output_initial)             # K * 4 channels
        bn1_out = self.bottleneck1_1(bn1_0)
        bn2_0 = self.bottleneck2_0(bn1_out)                     # K * 8 channels
        bn2_out = self.bottleneck2_1(bn2_0)

        # Middle bottleneck
        bn3_out = self.bottleneck3(bn2_out)                     # K * 8 channels

        # Decoder with Attention-Gated Skips
        bn4_up = self.bottleneck4(bn3_out, bn1_out)             # K * 4 channels
        bn4_out = self.bottleneck4_refine(bn4_up)               # K * 2 channels

        bn5_up = self.bottleneck5(bn4_out, output_initial)      # K channels
        bn5_out = self.bottleneck5_refine(bn5_up)               # K channels

        # Refine and classify main output
        feats = self.pre_final(bn5_out)
        interpolated = F.interpolate(
            feats, size=input.shape[2:], mode="trilinear", align_corners=False
        )
        main_logits = self.classifier(interpolated)

        # Deep Supervision output during training
        if self.training:
            aux_logits = self.aux_classifier(bn4_out)
            aux_logits = F.interpolate(
                aux_logits, size=input.shape[2:], mode="trilinear", align_corners=False
            )
            return main_logits, aux_logits

        return main_logits

    def init_weights(self, *args, **kwargs):
        self.apply(init_weights_3d)


class AttentionENet3D(ENet3D):
    """3D ENet with channel attention at the bottleneck feature scale + SE & Attention Skips."""

    def __init__(self, in_dim: int = 1, out_dim: int = 5, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        kernels = kwargs.get("kernels", 32)
        self.attention_down = ChannelAttention3D(kernels * 8)
        self.attention_middle = ChannelAttention3D(kernels * 8)

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        conv_0 = self.conv0(input)
        maxpool_0 = self.maxpool0(input)
        output_initial = torch.cat((conv_0, maxpool_0), dim=1)

        bn1_0 = self.bottleneck1_0(output_initial)
        bn1_out = self.bottleneck1_1(bn1_0)
        bn2_0 = self.bottleneck2_0(bn1_out)
        bn2_out = self.bottleneck2_1(bn2_0)

        bn2_out = self.attention_down(bn2_out)
        bn3_out = self.bottleneck3(bn2_out)
        bn3_out = self.attention_middle(bn3_out)

        bn4_up = self.bottleneck4(bn3_out, bn1_out)
        bn4_out = self.bottleneck4_refine(bn4_up)

        bn5_up = self.bottleneck5(bn4_out, output_initial)
        bn5_out = self.bottleneck5_refine(bn5_up)

        feats = self.pre_final(bn5_out)
        interpolated = F.interpolate(
            feats, size=input.shape[2:], mode="trilinear", align_corners=False
        )
        main_logits = self.classifier(interpolated)

        if self.training:
            aux_logits = self.aux_classifier(bn4_out)
            aux_logits = F.interpolate(
                aux_logits, size=input.shape[2:], mode="trilinear", align_corners=False
            )
            return main_logits, aux_logits

        return main_logits


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