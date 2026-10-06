#!/usr/bin/env python3

from typing import Tuple, Union
import math
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
    """Weight initialization matching Vision Transformer conventions."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =====================================================================
# Vision Transformer Modules
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


class SpatialAttention2D(nn.Module):
    """Spatial Reduction Attention (SRA) for high-resolution feature maps."""

    def __init__(self, dim: int, num_heads: int = 4, sr_ratio: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = get_group_norm(dim)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.flatten(2).transpose(1, 2)

        # Query uses full spatial resolution
        q = (
            self.q(x_flat)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Key & Value are spatially downsampled if sr_ratio > 1
        if self.sr_ratio > 1:
            x_sr = self.norm(self.sr(x)).flatten(2).transpose(1, 2)
            kv = (
                self.kv(x_sr)
                .reshape(B, -1, 2, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x_flat)
                .reshape(B, N, 2, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )

        k, v = kv[0], kv[1]

        # Memory-efficient attention
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out.transpose(1, 2).reshape(B, C, H, W)


class MLP2D(nn.Module):
    """Feed-Forward Network (FFN) for 2D Spatial Features."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ViTBlock(nn.Module):
    """Vision Transformer Block replacing ConvNeXt Block with MHSA + FeedForward MLP."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int = 1,
        expand_ratio: int = 4,
        num_heads: int = 4,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        hidden_dim = in_dim * expand_ratio

        # Downsampling / Projection Layer
        if stride != 1 or in_dim != out_dim:
            self.downsample = nn.Conv2d(
                in_dim, out_dim, kernel_size=stride, stride=stride, bias=False
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                get_group_norm(out_dim),
            )
            current_dim = out_dim
        else:
            self.downsample = nn.Identity()
            self.shortcut = nn.Identity()
            current_dim = in_dim

        # Ensure num_heads divides channel dimension
        num_heads = min(num_heads, current_dim)
        while current_dim % num_heads != 0:
            num_heads -= 1

        self.norm1 = get_group_norm(current_dim)
        self.attn = SpatialAttention2D(current_dim, num_heads=num_heads)

        self.norm2 = get_group_norm(current_dim)
        self.mlp = MLP2D(current_dim, hidden_dim, current_dim)
        self.se = SqueezeExcite(current_dim)

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        input_tensor = x
        x = self.downsample(x)

        # Self-Attention Branch
        res = self.attn(self.norm1(x))
        x = x + self.drop_path(res)

        # FFN Branch
        res = self.se(self.mlp(self.norm2(x)))
        x = x + self.drop_path(res)

        return (
            self.shortcut(input_tensor)
            if isinstance(self.downsample, nn.Identity)
            else x
        )


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
    """Attention-Gated Decoder Block with Skip Fusion and Transformer Refinement."""

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

        self.spatial = ViTBlock(
            out_dim, out_dim, stride=1, drop_path_rate=drop_path_rate
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x_up = F.interpolate(
            x, size=skip.shape[2:], mode="bilinear", align_corners=False
        )
        skip_gated = self.attn_gate(g=x_up, x=skip)
        reduced = self.reduce(torch.cat([x_up, skip_gated], dim=1))
        return self.spatial(reduced)


# =====================================================================
# Main Architecture: ViT_ImprovedENet
# =====================================================================


class ViT(nn.Module):
    """Vision Transformer (ViT-UNet) counterpart to ImprovedENet using 2D MHSA blocks."""

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
            ViTBlock(K, K, stride=1, drop_path_rate=0.0),
        )

        # Encoder Path (Hierarchical Vision Transformer)
        self.enc1 = ViTBlock(
            K, K * 2, stride=2, num_heads=2, drop_path_rate=drop_rate * 0.2
        )

        self.enc2 = nn.Sequential(
            ViTBlock(
                K * 2, K * 4, stride=2, num_heads=4, drop_path_rate=drop_rate * 0.4
            ),
            ViTBlock(
                K * 4, K * 4, stride=1, num_heads=4, drop_path_rate=drop_rate * 0.4
            ),
        )

        self.enc3 = nn.Sequential(
            ViTBlock(
                K * 4, K * 8, stride=2, num_heads=8, drop_path_rate=drop_rate * 0.6
            ),
            ViTBlock(
                K * 8, K * 8, stride=1, num_heads=8, drop_path_rate=drop_rate * 0.6
            ),
        )

        # Bottleneck (ASPP + Global ViT Attention)
        self.bottleneck = nn.Sequential(
            LiteDepthwiseASPPModule(K * 8, K * 8, rates=(1, 3, 6)),
            ViTBlock(
                K * 8, K * 8, stride=1, num_heads=8, drop_path_rate=bottleneck_drop_rate
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
