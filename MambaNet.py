#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from mamba_ssm import Mamba3 as Mamba
except ImportError:
    Mamba = None


def get_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Dynamic GroupNorm matching the working ImprovedENet setup."""
    for g in [32, 16, 8, 4, 2]:
        if g <= max_groups and num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """Stochastic Depth / DropPath implementation."""
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
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class SqueezeExcite(nn.Module):
    """Channel attention module."""

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
    """ConvNeXt-style 7x7 Depthwise Separable Block with SqueezeExcite."""

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
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            get_group_norm(out_dim),
            nn.SiLU(inplace=True),
        )
        self.se = SqueezeExcite(out_dim)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                get_group_norm(out_dim),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.drop_path(self.se(self.conv(x)))


class LiteDepthwiseASPPModule(nn.Module):
    """Lite Atrous Spatial Pyramid Pooling for lightweight multi-scale context."""

    def __init__(self, in_dim: int, out_dim: int, rates: tuple = (1, 3, 6)):
        super().__init__()
        mid_dim = max(16, in_dim // 4)
        self.branches = nn.ModuleList()

        # 1x1 Conv Branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_dim, mid_dim, kernel_size=1, bias=False),
                get_group_norm(mid_dim),
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
                    nn.Conv2d(in_dim, mid_dim, kernel_size=1, bias=False),
                    get_group_norm(mid_dim),
                    nn.SiLU(inplace=True),
                )
            )

        # Global Pooling Branch
        self.glob_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, bias=False),
            get_group_norm(mid_dim),
            nn.SiLU(inplace=True),
        )

        num_branches = len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(mid_dim * num_branches, out_dim, kernel_size=1, bias=False),
            get_group_norm(out_dim),
            nn.SiLU(inplace=True),
            SqueezeExcite(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]
        res = [branch(x) for branch in self.branches]

        gp = self.glob_pool(x)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        res.append(gp)

        return self.project(torch.cat(res, dim=1))


class Mamba2DBlock(nn.Module):
    """Bidirectional Spatial SSM block with DropPath regularization."""

    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        expand: int = 1,
        headdim: int = 8,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        if Mamba is None:
            raise ImportError("mamba_ssm package is not installed.")

        d_inner = dim * expand
        if d_inner % headdim != 0:
            headdim = d_inner

        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            is_mimo=False,
        )
        self.proj = nn.Linear(dim, dim)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_flat)

        # Bidirectional scan reuse
        out_fw = self.mamba(x_norm)
        out_bw = torch.flip(self.mamba(torch.flip(x_norm, dims=[1])), dims=[1])

        mamba_out = self.proj(out_fw + out_bw)
        out = x_flat + self.drop_path(mamba_out)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class UpDecoderBlock(nn.Module):
    """Refined Decoder Block with Skip Fusion and optional Mamba scanning."""

    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        out_dim: int,
        use_mamba: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        concat_dim = in_dim + skip_dim
        self.reduce = nn.Sequential(
            nn.Conv2d(concat_dim, out_dim, kernel_size=1, bias=False),
            get_group_norm(out_dim),
            nn.SiLU(inplace=True),
        )

        if use_mamba:
            self.spatial = Mamba2DBlock(dim=out_dim, drop_path_rate=drop_path_rate)
        else:
            self.spatial = DepthwiseSeparableBlock(
                out_dim, out_dim, kernel_size=7, drop_path_rate=drop_path_rate
            )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x_up = F.interpolate(
            x, size=skip.shape[2:], mode="bilinear", align_corners=False
        )
        reduced = self.reduce(torch.cat([x_up, skip], dim=1))
        return self.spatial(reduced)


class MambaNet(nn.Module):
    """Upgraded Hybrid Mamba-ConvNeXt Architecture (~350k parameters)."""

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        drop_rate: float = 0.1,
        bottleneck_drop_rate: float = 0.2,
        **kwargs
    ):
        super().__init__()
        K: int = kwargs.get("kernels", 16)  # Base channels (16 -> 32 -> 64 -> 128)

        # Stem (256x256)
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, K, kernel_size=3, padding=1, bias=False),
            get_group_norm(K),
            nn.SiLU(inplace=True),
            DepthwiseSeparableBlock(K, K, kernel_size=3, drop_path_rate=0.0),
        )

        # Encoder Path
        # Stage 1: 256x256 -> 128x128
        self.enc1 = DepthwiseSeparableBlock(
            K, K * 2, kernel_size=7, stride=2, drop_path_rate=drop_rate * 0.2
        )

        # Stage 2: 128x128 -> 64x64 (Mamba-enabled)
        self.enc2 = nn.Sequential(
            DepthwiseSeparableBlock(
                K * 2, K * 4, kernel_size=7, stride=2, drop_path_rate=drop_rate * 0.4
            ),
            Mamba2DBlock(dim=K * 4, drop_path_rate=drop_rate * 0.4),
        )

        # Stage 3: 64x64 -> 32x32 (Mamba-enabled)
        self.enc3 = nn.Sequential(
            DepthwiseSeparableBlock(
                K * 4, K * 8, kernel_size=7, stride=2, drop_path_rate=drop_rate * 0.6
            ),
            Mamba2DBlock(dim=K * 8, drop_path_rate=drop_rate * 0.6),
        )

        # Bottleneck (32x32): Lite-ASPP + Mamba
        self.bottleneck = nn.Sequential(
            LiteDepthwiseASPPModule(K * 8, K * 8, rates=(1, 3, 6)),
            Mamba2DBlock(dim=K * 8, drop_path_rate=bottleneck_drop_rate),
        )

        # Decoder Path
        self.dec3 = UpDecoderBlock(
            in_dim=K * 8,
            skip_dim=K * 4,
            out_dim=K * 4,
            use_mamba=True,
            drop_path_rate=drop_rate * 0.4,
        )
        self.dec2 = UpDecoderBlock(
            in_dim=K * 4,
            skip_dim=K * 2,
            out_dim=K * 2,
            use_mamba=False,
            drop_path_rate=drop_rate * 0.2,
        )
        self.dec1 = UpDecoderBlock(
            in_dim=K * 2,
            skip_dim=K,
            out_dim=K,
            use_mamba=False,
            drop_path_rate=0.0,
        )

        # Final Classifier
        self.final = nn.Conv2d(K, out_dim, kernel_size=1)

        self.apply(random_weights_init)

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
