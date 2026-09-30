#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Union

try:
    from mamba_ssm import Mamba3 as Mamba
except ImportError:
    Mamba = None

# =====================================================================
# Utilities & Normalization
# =====================================================================


def get_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Dynamic GroupNorm matching channel divisibility."""
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


def model_weights_init(m: nn.Module) -> None:
    """Weight initialization for ConvNeXt & SSM modules."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
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
    """ConvNeXt Inverted Bottleneck Block with LayerScale."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 7,
        stride: int = 1,
        expand_ratio: int = 4,
        drop_path_rate: float = 0.0,
        layer_scale_init: float = 1e-6,
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

        self.gamma = (
            nn.Parameter(
                layer_scale_init * torch.ones(out_dim, 1, 1), requires_grad=True
            )
            if layer_scale_init > 0
            else None
        )

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

        if self.gamma is not None:
            x = x * self.gamma

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
    """Lite Atrous Spatial Pyramid Pooling module."""

    def __init__(self, in_dim: int, out_dim: int, rates: tuple = (1, 3, 6)):
        super().__init__()
        mid_dim = max(16, in_dim // 4)
        self.branches = nn.ModuleList()

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_dim, mid_dim, kernel_size=1, bias=False),
                get_group_norm(mid_dim),
                nn.SiLU(inplace=True),
            )
        )

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


# =====================================================================
# Optimized 2D Cross-Scan Mamba Module
# =====================================================================


class Mamba2DBlock(nn.Module):
    """Batched 4-Directional 2D Spatial Cross-Scanning SSM Block.

    Executes all 4 spatial scanning directions in a SINGLE batched SSM kernel
    pass [4*B, L, C] to maximize GPU occupancy and eliminate launch overhead.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        expand: int = 1,
        headdim: int = 8,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

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

        self.proj = nn.Linear(dim * 4, dim)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    @torch.compiler.disable()
    def mamba_call(self, x: Tensor) -> Tensor:
        return self.mamba(x)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        L = H * W

        # Permute to (B, H, W, C) & Normalize
        x_norm = self.norm(x.permute(0, 2, 3, 1).contiguous())

        # Construct 4 directional scan sequences
        # 1. Horizontal Forward
        x1 = x_norm.view(B, L, C)
        # 2. Horizontal Backward
        x2 = torch.flip(x1, dims=[1])
        # 3. Vertical Forward
        x3 = x_norm.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        # 4. Vertical Backward
        x4 = torch.flip(x3, dims=[1])

        # Batch all 4 directions together [4*B, L, C] for a single kernel pass
        xs = torch.cat([x1, x2, x3, x4], dim=0)

        ys = self.mamba_call(xs)

        # Split outputs back to 4 streams
        y1, y2, y3, y4 = torch.chunk(ys, 4, dim=0)

        # Unflip and re-arrange back to 2D image spatial structure
        y1 = y1.view(B, H, W, C)
        y2 = torch.flip(y2, dims=[1]).view(B, H, W, C)
        y3 = y3.view(B, W, H, C).permute(0, 2, 1, 3)
        y4 = torch.flip(y4, dims=[1]).view(B, W, H, C).permute(0, 2, 1, 3)

        # Concatenate 4 scan features and project back to original channel dim
        merged = torch.cat([y1, y2, y3, y4], dim=-1)
        mamba_out = self.proj(merged).permute(0, 3, 1, 2)

        return x + self.drop_path(mamba_out)


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
        use_mamba: bool = False,
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

        if use_mamba:
            self.spatial = Mamba2DBlock(dim=out_dim, drop_path_rate=drop_path_rate)
        else:
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
# Main Architecture: MambaNet
# =====================================================================


class MambaNet(nn.Module):
    """Hybrid Mamba-ConvNeXt Architecture with Batched 2D Cross-Scan."""

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
            Mamba2DBlock(dim=K * 4, drop_path_rate=drop_rate * 0.4),
        )

        self.enc3 = nn.Sequential(
            ConvNeXtBlock(
                K * 4, K * 8, kernel_size=7, stride=2, drop_path_rate=drop_rate * 0.6
            ),
            Mamba2DBlock(dim=K * 8, drop_path_rate=drop_rate * 0.6),
        )

        # Bottleneck: Lite-ASPP + 2D Mamba
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

        self.apply(model_weights_init)

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
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

        out = self.final(d1)
        return out

    def init_weights(self):
        self.apply(model_weights_init)
