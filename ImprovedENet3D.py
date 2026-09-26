#!/usr/bin/env python3.10

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def random_weights_init_3d(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class SqueezeExcite3D(nn.Module):
    """3D Channel attention to highlight small volumetric structures."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, reduced, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.fc(x)


class DepthwiseSeparableBlock3D(nn.Module):
    """Stable 3D block utilizing GroupNorm for small 3D batch sizes."""

    def __init__(self, in_dim: int, out_dim: int, stride: int = 1):
        super().__init__()
        groups_in = 8 if in_dim % 8 == 0 else 1
        groups_out = 8 if out_dim % 8 == 0 else 1

        self.conv = nn.Sequential(
            # 3D Depthwise
            nn.Conv3d(
                in_dim,
                in_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_dim,
                bias=False,
            ),
            nn.GroupNorm(groups_in, in_dim),
            nn.SiLU(inplace=True),
            # 3D Pointwise
            nn.Conv3d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups_out, out_dim),
            nn.SiLU(inplace=True),
        )
        self.se = SqueezeExcite3D(out_dim)

        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups_out, out_dim),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.se(self.conv(x)) + self.shortcut(x)


class UpDecoderBlock3D(nn.Module):
    """3D UpDecoder Block using Trilinear Interpolation."""

    def __init__(self, in_dim: int, skip_dim: int, out_dim: int):
        super().__init__()
        self.conv = DepthwiseSeparableBlock3D(in_dim + skip_dim, out_dim)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x_up = F.interpolate(
            x, size=skip.shape[2:], mode="trilinear", align_corners=False
        )
        fused = torch.cat([x_up, skip], dim=1)
        return self.conv(fused)


class ImprovedENet3D(nn.Module):
    """Full 3D ImprovedENet Architecture for volumetric segmentation [B, C, Z, H, W]."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        factor: int = kwargs.get("factor", 2)
        K: int = max(16, kwargs.get("kernels", 16) * factor)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_dim, K, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if K % 8 == 0 else 1, K),
            nn.SiLU(inplace=True),
            DepthwiseSeparableBlock3D(K, K),
        )

        # Encoder
        self.enc1 = DepthwiseSeparableBlock3D(K, K * 2, stride=2)
        self.enc2 = DepthwiseSeparableBlock3D(K * 2, K * 4, stride=2)
        self.enc3 = DepthwiseSeparableBlock3D(K * 4, K * 8, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableBlock3D(K * 8, K * 8),
            DepthwiseSeparableBlock3D(K * 8, K * 8),
            DepthwiseSeparableBlock3D(K * 8, K * 8),
        )

        # Decoder
        self.dec3 = UpDecoderBlock3D(in_dim=K * 8, skip_dim=K * 4, out_dim=K * 4)
        self.dec2 = UpDecoderBlock3D(in_dim=K * 4, skip_dim=K * 2, out_dim=K * 2)
        self.dec1 = UpDecoderBlock3D(in_dim=K * 2, skip_dim=K, out_dim=K)

        # Final Classifier
        self.final = nn.Conv3d(K, out_dim, kernel_size=1)

        print(
            f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}"
        )

    def forward(self, input: Tensor) -> Tensor:
        x0 = self.stem(input)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        b = self.bottleneck(x3)

        d3 = self.dec3(b, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        return self.final(d1)

    def init_weights(self, *args, **kwargs):
        self.apply(random_weights_init_3d)
