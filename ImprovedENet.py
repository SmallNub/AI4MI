#!/usr/bin/env python3

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# =====================================================================
# Utilities & Normalization
# =====================================================================


def get_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Dynamic GroupNorm matching channel divisibility."""
    effective_max = min(max_groups, max(1, num_channels // 2))
    for g in [32, 16, 8, 4, 2]:
        if g <= effective_max and num_channels % g == 0:
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


def model_weights_init(m: nn.Module) -> None:
    """Weight initialization matching MambaNet."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =====================================================================
# ConvNeXt & Attention Modules
# =====================================================================


class SqueezeExcite(nn.Module):
    """Channel attention with dynamic reduction."""

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


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Inverted Bottleneck Block with GroupNorm."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 7,
        stride: int = 1,
        expand_ratio: int = 4,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2
        hidden_dim = in_dim * expand_ratio

        self.dwconv = nn.Conv2d(
            in_dim,
            in_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_dim,
            bias=False,
        )
        self.norm = get_group_norm(in_dim)
        self.pwconv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.pwconv2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False)
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
        input_tensor = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.se(x)

        return self.shortcut(input_tensor) + self.drop_path(x)


class SpatialAttentionGate(nn.Module):
    """Attention Gate for skip feature filtering before decoder fusion."""

    def __init__(self, gate_dim: int, skip_dim: int, inter_dim: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_dim, inter_dim, kernel_size=1, bias=False),
            get_group_norm(inter_dim),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_dim, inter_dim, kernel_size=1, bias=False),
            get_group_norm(inter_dim),
        )
        self.psi = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        alpha = self.psi(g1 + x1)
        return x * alpha


class LiteDepthwiseASPPModule(nn.Module):
    """Lite Atrous Spatial Pyramid Pooling module using additive feature fusion."""

    def __init__(self, in_dim: int, out_dim: int, rates: tuple = (1, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList()

        # 1x1 conv branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                get_group_norm(out_dim),
                nn.SiLU(inplace=True),
            )
        )

        # Dilated depthwise conv branches
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

        # Global average pooling branch
        self.glob_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            get_group_norm(out_dim),
            nn.SiLU(inplace=True),
        )

        # Final projection & channel attention
        self.project = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            get_group_norm(out_dim),
            nn.SiLU(inplace=True),
            SqueezeExcite(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]

        # Collect branch outputs
        fused = self.branches[0](x)
        for branch in self.branches[1:]:
            fused = fused + branch(x)

        gp = self.glob_pool(x)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        fused = fused + gp

        return self.project(fused)


# =====================================================================
# Decoder Block
# =====================================================================


class UpDecoderBlock(nn.Module):
    """Attention-Gated Decoder Block with Skip Fusion."""

    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        out_dim: int,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.attn_gate = SpatialAttentionGate(
            gate_dim=in_dim, skip_dim=skip_dim, inter_dim=out_dim // 2
        )

        concat_dim = in_dim + skip_dim
        self.reduce = nn.Sequential(
            nn.Conv2d(concat_dim, out_dim, kernel_size=1, bias=False),
            get_group_norm(out_dim),
            nn.SiLU(inplace=True),
        )

        self.spatial = ConvNeXtBlock(
            out_dim, out_dim, kernel_size=7, drop_path_rate=drop_path_rate
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x_up = F.interpolate(
            x, size=skip.shape[2:], mode="bilinear", align_corners=False
        )
        skip_gated = self.attn_gate(g=x_up, x=skip)
        reduced = self.reduce(torch.cat([x_up, skip_gated], dim=1))
        return self.spatial(reduced)


# =====================================================================
# Main Architecture: ImprovedENet
# =====================================================================


class ImprovedENet(nn.Module):
    """Pure CNN counterpart to MambaNet using ConvNeXt blocks in place of Mamba2D."""

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        drop_rate: float = 0.1,
        bottleneck_drop_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        K: int = kwargs.get("kernels", 16)

        # Stem Layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, K, kernel_size=3, padding=1, bias=False),
            get_group_norm(K),
            nn.SiLU(inplace=True),
            ConvNeXtBlock(K, K, kernel_size=3, drop_path_rate=0.0),
        )

        # Encoder Path
        self.enc1 = ConvNeXtBlock(
            K, K * 2, kernel_size=7, stride=2, drop_path_rate=drop_rate * 0.2
        )

        self.enc2 = nn.Sequential(
            ConvNeXtBlock(
                K * 2, K * 4, kernel_size=7, stride=2, drop_path_rate=drop_rate * 0.4
            ),
            ConvNeXtBlock(
                K * 4, K * 4, kernel_size=7, stride=1, drop_path_rate=drop_rate * 0.4
            ),
        )

        self.enc3 = nn.Sequential(
            ConvNeXtBlock(
                K * 4, K * 8, kernel_size=7, stride=2, drop_path_rate=drop_rate * 0.6
            ),
            ConvNeXtBlock(
                K * 8, K * 8, kernel_size=7, stride=1, drop_path_rate=drop_rate * 0.6
            ),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            LiteDepthwiseASPPModule(K * 8, K * 8, rates=(1, 3, 6)),
            ConvNeXtBlock(
                K * 8, K * 8, kernel_size=7, drop_path_rate=bottleneck_drop_rate
            ),
        )

        # Decoder Path
        self.dec3 = UpDecoderBlock(
            in_dim=K * 8,
            skip_dim=K * 4,
            out_dim=K * 4,
            drop_path_rate=drop_rate * 0.4,
        )
        self.dec2 = UpDecoderBlock(
            in_dim=K * 4,
            skip_dim=K * 2,
            out_dim=K * 2,
            drop_path_rate=drop_rate * 0.2,
        )
        self.dec1 = UpDecoderBlock(
            in_dim=K * 2,
            skip_dim=K,
            out_dim=K,
            drop_path_rate=0.0,
        )

        # Final Classifier
        self.final = nn.Conv2d(K, out_dim, kernel_size=1)

        self.apply(model_weights_init)

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        x0 = self.stem(input)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        b = self.bottleneck(x3)

        d3 = self.dec3(b, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        out = self.final(d1)
        return out

    def init_weights(self):
        self.apply(model_weights_init)
