#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_group_norm_3d(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Helper to dynamically choose a valid number of groups for GroupNorm (3D)."""
    for g in [32, 16, 8, 4, 2]:
        if g <= max_groups and num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


def random_weights_init_3d(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class DropPath3D(nn.Module):
    """Stochastic Depth (DropPath) per sample for 3D residual structures."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class SqueezeExcite3D(nn.Module):
    """3D Channel attention optimized for single-channel/sparse volumetric inputs."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(4, channels // reduction)
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
    """Enhanced 3D Depthwise Separable Block using DropPath instead of aggressive Dropout3d."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int | tuple[int, int, int] = 1,
        kernel_size: int | tuple[int, int, int] = 3,
        dilation: int | tuple[int, int, int] = 1,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        padding = tuple((k - 1) // 2 * d for k, d in zip(kernel_size, dilation))

        self.conv = nn.Sequential(
            # Depthwise 3D
            nn.Conv3d(
                in_dim,
                in_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_dim,
                bias=False,
            ),
            get_group_norm_3d(in_dim),
            nn.SiLU(inplace=True),
            # Pointwise 3D
            nn.Conv3d(in_dim, out_dim, kernel_size=1, bias=False),
            get_group_norm_3d(out_dim),
            nn.SiLU(inplace=True),
        )
        self.se = SqueezeExcite3D(out_dim)
        self.drop_path = DropPath3D(drop_rate) if drop_rate > 0.0 else nn.Identity()

        if stride != 1 or stride != (1, 1, 1) or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                get_group_norm_3d(out_dim),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.drop_path(self.se(self.conv(x)))


class UpDecoderBlock3D(nn.Module):
    """3D Decoder Block using trilinear upsampling with residual shortcut."""

    def __init__(
        self, in_dim: int, skip_dim: int, out_dim: int, drop_rate: float = 0.0
    ):
        super().__init__()
        self.conv = DepthwiseSeparableBlock3D(
            in_dim + skip_dim, out_dim, drop_rate=drop_rate
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x_up = F.interpolate(
            x, size=skip.shape[2:], mode="trilinear", align_corners=False
        )
        fused = torch.cat([x_up, skip], dim=1)
        return self.conv(fused)


class ImprovedENet3D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        drop_rate: float = 0.1,
        bottleneck_drop_rate: float = 0.2,
        **kwargs
    ):
        super().__init__()
        factor: int = kwargs.get("factor", 2)
        K: int = max(16, kwargs.get("kernels", 16) * factor)

        # Stem: Handles single-channel input without aggressive channel dropout
        self.stem = nn.Sequential(
            nn.Conv3d(in_dim, K, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            get_group_norm_3d(K),
            nn.SiLU(inplace=True),
            DepthwiseSeparableBlock3D(K, K, drop_rate=0.0),
        )

        # Encoders with anisotropic Z-handling for medical scans
        self.enc1 = DepthwiseSeparableBlock3D(K, K * 2, stride=(1, 2, 2), drop_rate=drop_rate)
        self.enc2 = DepthwiseSeparableBlock3D(K * 2, K * 4, stride=2, drop_rate=drop_rate)
        self.enc3 = DepthwiseSeparableBlock3D(K * 4, K * 8, stride=2, drop_rate=drop_rate)

        # Bottleneck: Multi-scale 3D Dilations to collect context without spatial collapse
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableBlock3D(K * 8, K * 8, dilation=(1, 1, 1), drop_rate=bottleneck_drop_rate),
            DepthwiseSeparableBlock3D(K * 8, K * 8, dilation=(1, 2, 2), drop_rate=bottleneck_drop_rate),
            DepthwiseSeparableBlock3D(K * 8, K * 8, dilation=(2, 4, 4), drop_rate=bottleneck_drop_rate),
        )

        # Decoders
        self.dec3 = UpDecoderBlock3D(
            in_dim=K * 8, skip_dim=K * 4, out_dim=K * 4, drop_rate=drop_rate
        )
        self.dec2 = UpDecoderBlock3D(
            in_dim=K * 4, skip_dim=K * 2, out_dim=K * 2, drop_rate=drop_rate
        )
        self.dec1 = UpDecoderBlock3D(
            in_dim=K * 2, skip_dim=K, out_dim=K, drop_rate=drop_rate
        )

        # Final Classifier
        self.final = nn.Conv3d(K, out_dim, kernel_size=1)

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
