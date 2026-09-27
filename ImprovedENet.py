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


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """Stochastic Depth / DropPath implementation for residual connections."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


def random_weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class SqueezeExcite(nn.Module):
    """Channel attention mechanism."""

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
    """ConvNeXt-style Large Kernel (7x7) Depthwise Separable Block with Stochastic Depth."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 7,
        stride: int = 1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            # Depthwise Large Kernel
            nn.Conv2d(
                in_dim,
                in_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_dim,
                bias=False,
            ),
            get_group_norm(in_dim),
            nn.SiLU(inplace=True),
            # Pointwise expansion/projection
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            get_group_norm(out_dim),
            nn.SiLU(inplace=True),
        )
        self.se = SqueezeExcite(out_dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                get_group_norm(out_dim),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.drop_path(self.se(self.conv(x)))


class DepthwiseASPPModule(nn.Module):
    """Lightweight Atrous Spatial Pyramid Pooling using Depthwise Separable Convolutions."""

    def __init__(self, in_dim: int, out_dim: int, rates: tuple = (1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()

        # 1x1 Conv Branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                get_group_norm(out_dim),
                nn.SiLU(inplace=True),
            )
        )

        # Dilated Depthwise Separable Branches
        for rate in rates[1:]:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_dim,
                        in_dim,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        groups=in_dim,
                        bias=False,
                    ),
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                    get_group_norm(out_dim),
                    nn.SiLU(inplace=True),
                )
            )

        # Global Pooling Branch
        self.glob_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            get_group_norm(out_dim),
            nn.SiLU(inplace=True),
        )

        # Output projection
        num_branches = len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_dim * num_branches, out_dim, kernel_size=1, bias=False),
            get_group_norm(out_dim),
            nn.SiLU(inplace=True),
            SqueezeExcite(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]
        res = [branch(x) for branch in self.branches]

        # Global pooling branch with dynamic resize
        gp = self.glob_pool(x)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        res.append(gp)

        return self.project(torch.cat(res, dim=1))


class UpDecoderBlock(nn.Module):
    def __init__(
        self, in_dim: int, skip_dim: int, out_dim: int, drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.conv = DepthwiseSeparableBlock(
            in_dim + skip_dim, out_dim, kernel_size=7, drop_path_rate=drop_path_rate
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

        # Stem (3x3 depthwise for high-res edge extraction)
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, K, kernel_size=3, padding=1, bias=False),
            get_group_norm(K),
            nn.SiLU(inplace=True),
            DepthwiseSeparableBlock(K, K, kernel_size=3, drop_path_rate=0.0),
        )

        # Encoder (7x7 Large Kernels)
        self.enc1 = DepthwiseSeparableBlock(
            K, K * 2, kernel_size=7, stride=2, drop_path_rate=drop_rate * 0.25
        )
        self.enc2 = DepthwiseSeparableBlock(
            K * 2, K * 4, kernel_size=7, stride=2, drop_path_rate=drop_rate * 0.5
        )
        self.enc3 = DepthwiseSeparableBlock(
            K * 4, K * 8, kernel_size=7, stride=2, drop_path_rate=drop_rate * 0.75
        )

        # Bottleneck with DW-ASPP + Large Kernel refine
        self.bottleneck = nn.Sequential(
            DepthwiseASPPModule(K * 8, K * 8, rates=(1, 6, 12, 18)),
            DepthwiseSeparableBlock(
                K * 8, K * 8, kernel_size=7, drop_path_rate=bottleneck_drop_rate
            ),
        )

        # Decoder
        self.dec3 = UpDecoderBlock(
            in_dim=K * 8, skip_dim=K * 4, out_dim=K * 4, drop_path_rate=drop_rate * 0.5
        )
        self.dec2 = UpDecoderBlock(
            in_dim=K * 4, skip_dim=K * 2, out_dim=K * 2, drop_path_rate=drop_rate * 0.25
        )
        self.dec1 = UpDecoderBlock(
            in_dim=K * 2, skip_dim=K, out_dim=K, drop_path_rate=0.0
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
