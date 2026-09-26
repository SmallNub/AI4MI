#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Helper to dynamically choose a valid number of groups for GroupNorm."""
    for g in [32, 16, 8, 4, 2]:
        if g <= max_groups and num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


def random_weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class SqueezeExcite(nn.Module):
    """Channel attention to highlight small organs."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.fc(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, stride: int = 1, drop_rate: float = 0.0
    ):
        super().__init__()

        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(
                in_dim,
                in_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_dim,
                bias=False,
            ),
            get_group_norm(in_dim),
            nn.SiLU(inplace=True),
            # Spatial Dropout across channels
            nn.Dropout2d(p=drop_rate) if drop_rate > 0 else nn.Identity(),
            # Pointwise
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            get_group_norm(out_dim),
            nn.SiLU(inplace=True),
        )
        self.se = SqueezeExcite(out_dim)

        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                get_group_norm(out_dim),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.se(self.conv(x)) + self.shortcut(x)


class UpDecoderBlock(nn.Module):
    def __init__(
        self, in_dim: int, skip_dim: int, out_dim: int, drop_rate: float = 0.0
    ):
        super().__init__()
        self.conv = DepthwiseSeparableBlock(
            in_dim + skip_dim, out_dim, drop_rate=drop_rate
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x_up = F.interpolate(
            x, size=skip.shape[2:], mode="bilinear", align_corners=False
        )
        fused = torch.cat([x_up, skip], dim=1)
        return self.conv(fused)


class ImprovedENet(nn.Module):
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

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, K, kernel_size=3, padding=1, bias=False),
            get_group_norm(K),
            nn.SiLU(inplace=True),
            DepthwiseSeparableBlock(K, K, drop_rate=drop_rate),
        )

        # Encoder
        self.enc1 = DepthwiseSeparableBlock(K, K * 2, stride=2, drop_rate=drop_rate)
        self.enc2 = DepthwiseSeparableBlock(K * 2, K * 4, stride=2, drop_rate=drop_rate)
        self.enc3 = DepthwiseSeparableBlock(K * 4, K * 8, stride=2, drop_rate=drop_rate)

        # Bottleneck (slightly higher dropout for regularization)
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableBlock(K * 8, K * 8, drop_rate=bottleneck_drop_rate),
            DepthwiseSeparableBlock(K * 8, K * 8, drop_rate=bottleneck_drop_rate),
            DepthwiseSeparableBlock(K * 8, K * 8, drop_rate=bottleneck_drop_rate),
        )

        # Decoder
        self.dec3 = UpDecoderBlock(
            in_dim=K * 8, skip_dim=K * 4, out_dim=K * 4, drop_rate=drop_rate
        )
        self.dec2 = UpDecoderBlock(
            in_dim=K * 4, skip_dim=K * 2, out_dim=K * 2, drop_rate=drop_rate
        )
        self.dec1 = UpDecoderBlock(
            in_dim=K * 2, skip_dim=K, out_dim=K, drop_rate=drop_rate
        )

        # Final Classifier
        self.final = nn.Conv2d(K, out_dim, kernel_size=1)

    def forward(self, input: Tensor) -> Tensor:
        if input.dim() == 5:
            B, Z, C, H, W = input.shape
            input = input.view(B, Z * C, H, W)

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
        self.apply(random_weights_init)
