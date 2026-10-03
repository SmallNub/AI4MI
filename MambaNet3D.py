#!/usr/bin/env python3

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

try:
    from mamba_ssm import Mamba3 as Mamba
except ImportError:
    Mamba = None

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


def model_weights_init_3d(m: nn.Module) -> None:
    """Weight initialization for 3D ConvNeXt & SSM modules."""
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# =====================================================================
# Memory-Optimized 3D ConvNeXt & Attention Modules
# =====================================================================


class SqueezeExcite3D(nn.Module):
    """3D Channel attention with dynamic reduction."""

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


class ConvNeXtBlock3D(nn.Module):
    """3D ConvNeXt Inverted Bottleneck Block."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 7,
        stride: int = 1,
        expand_ratio: int = 2,  # Reduced default expansion ratio from 4 to 2 to save VRAM
        drop_path_rate: float = 0.0,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        padding = kernel_size // 2
        hidden_dim = in_dim * expand_ratio

        self.dwconv = nn.Conv3d(
            in_dim,
            in_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_dim,
            bias=False,
        )
        self.norm = get_group_norm(in_dim)
        self.pwconv1 = nn.Conv3d(in_dim, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.pwconv2 = nn.Conv3d(hidden_dim, out_dim, kernel_size=1, bias=False)
        self.se = SqueezeExcite3D(out_dim)

        self.gamma = (
            nn.Parameter(
                layer_scale_init * torch.ones(out_dim, 1, 1, 1),
                requires_grad=True,
            )
            if layer_scale_init > 0
            else None
        )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
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


class SpatialAttentionGate3D(nn.Module):
    """3D Attention Gate for skip feature filtering before decoder fusion."""

    def __init__(self, gate_dim: int, skip_dim: int, inter_dim: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_dim, inter_dim, kernel_size=1, bias=False),
            get_group_norm(inter_dim),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_dim, inter_dim, kernel_size=1, bias=False),
            get_group_norm(inter_dim),
        )
        self.psi = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv3d(inter_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        alpha = self.psi(g1 + x1)
        return x * alpha


class LiteDepthwiseASPPModule3D(nn.Module):
    """Memory-efficient 3D Atrous Spatial Pyramid Pooling module using additive fusion."""

    def __init__(self, in_dim: int, out_dim: int, rates: tuple = (1, 3, 6)):
        super().__init__()
        mid_dim = max(16, in_dim // 4)
        self.branches = nn.ModuleList()

        self.branches.append(
            nn.Sequential(
                nn.Conv3d(in_dim, mid_dim, kernel_size=1, bias=False),
                get_group_norm(mid_dim),
                nn.SiLU(inplace=True),
            )
        )

        for rate in rates[1:]:
            self.branches.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_dim,
                        in_dim,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        groups=in_dim,
                        bias=False,
                    ),
                    nn.Conv3d(in_dim, mid_dim, kernel_size=1, bias=False),
                    get_group_norm(mid_dim),
                    nn.SiLU(inplace=True),
                )
            )

        self.glob_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_dim, mid_dim, kernel_size=1, bias=False),
            get_group_norm(mid_dim),
            nn.SiLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv3d(mid_dim, out_dim, kernel_size=1, bias=False),
            get_group_norm(out_dim),
            nn.SiLU(inplace=True),
            SqueezeExcite3D(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        d, h, w = x.shape[2:]
        # Accumulate branch outputs directly to avoid allocating a large concatenated tensor
        fused = self.branches[0](x)
        for branch in self.branches[1:]:
            fused = fused + branch(x)

        gp = self.glob_pool(x)
        gp = F.interpolate(gp, size=(d, h, w), mode="trilinear", align_corners=False)
        fused = fused + gp

        return self.project(fused)


# =====================================================================
# Optimized 3D Cross-Scan Mamba Module
# =====================================================================


class Mamba3DBlock(nn.Module):
    """Batched 6-Directional 3D Volumetric Cross-Scanning SSM Block with Additive Fusion.

    Eliminates high-dimensional 6x channel concatenation to cut peak activation memory by ~80%.
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

        # Lightweight directional weighting instead of 6*dim projection layer
        self.dir_weights = nn.Parameter(torch.ones(6) / 6.0)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    @torch.compiler.disable()
    def mamba_call(self, x: Tensor) -> Tensor:
        return self.mamba(x)

    def forward(self, x: Tensor) -> Tensor:
        B, C, D, H, W = x.shape
        L = D * H * W

        x_norm = self.norm(x.permute(0, 2, 3, 4, 1).contiguous())

        # Construct 6 directional scan sequences
        x1 = x_norm.view(B, L, C)
        x2 = torch.flip(x1, dims=[1])

        x3 = x_norm.permute(0, 3, 4, 2, 1).contiguous().view(B, L, C)
        x4 = torch.flip(x3, dims=[1])

        x5 = x_norm.permute(0, 4, 2, 3, 1).contiguous().view(B, L, C)
        x6 = torch.flip(x5, dims=[1])

        # Batch 6 directions into a single SSM pass
        xs = torch.cat([x1, x2, x3, x4, x5, x6], dim=0)
        ys = self.mamba_call(xs)

        y1, y2, y3, y4, y5, y6 = torch.chunk(ys, 6, dim=0)

        # Unflip and re-arrange back to 3D volumetric spatial structure
        y1 = y1.view(B, D, H, W, C)
        y2 = torch.flip(y2, dims=[1]).view(B, D, H, W, C)

        y3 = y3.view(B, H, W, D, C).permute(0, 3, 1, 2, 4)
        y4 = torch.flip(y4, dims=[1]).view(B, H, W, D, C).permute(0, 3, 1, 2, 4)

        y5 = y5.view(B, W, D, H, C).permute(0, 2, 3, 1, 4)
        y6 = torch.flip(y6, dims=[1]).view(B, W, D, H, C).permute(0, 2, 3, 1, 4)

        # Weighted additive directional aggregation (Avoids creating a 6*C intermediate tensor)
        w = F.softmax(self.dir_weights, dim=0)
        merged = w[0] * y1 + w[1] * y2 + w[2] * y3 + w[3] * y4 + w[4] * y5 + w[5] * y6

        mamba_out = merged.permute(0, 4, 1, 2, 3)
        return x + self.drop_path(mamba_out)


# =====================================================================
# 3D Decoder Block
# =====================================================================


class UpDecoderBlock3D(nn.Module):
    """3D Attention-Gated Decoder Block with Skip Fusion."""

    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        out_dim: int,
        use_mamba: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.attn_gate = SpatialAttentionGate3D(
            gate_dim=in_dim, skip_dim=skip_dim, inter_dim=out_dim // 2
        )

        concat_dim = in_dim + skip_dim
        self.reduce = nn.Sequential(
            nn.Conv3d(concat_dim, out_dim, kernel_size=1, bias=False),
            get_group_norm(out_dim),
            nn.SiLU(inplace=True),
        )

        if use_mamba:
            self.spatial = Mamba3DBlock(
                dim=out_dim,
                drop_path_rate=drop_path_rate,
            )
        else:
            self.spatial = ConvNeXtBlock3D(
                out_dim,
                out_dim,
                kernel_size=7,
                drop_path_rate=drop_path_rate,
            )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x_up = F.interpolate(
            x, size=skip.shape[2:], mode="trilinear", align_corners=False
        )
        skip_gated = self.attn_gate(g=x_up, x=skip)
        reduced = self.reduce(torch.cat([x_up, skip_gated], dim=1))
        return self.spatial(reduced)


# =====================================================================
# Main Architecture: MambaNet3D
# =====================================================================


class MambaNet3D(nn.Module):
    """Hybrid Mamba-ConvNeXt 3D Volumetric Architecture [B, C, D, H, W]."""

    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 1,
        drop_rate: float = 0.1,
        bottleneck_drop_rate: float = 0.2,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        K: int = kwargs.get("kernels", 16)

        # 3D Stem Layer
        self.stem = nn.Sequential(
            nn.Conv3d(in_dim, K, kernel_size=3, padding=1, bias=False),
            get_group_norm(K),
            nn.SiLU(inplace=True),
            ConvNeXtBlock3D(K, K, kernel_size=3, drop_path_rate=0.0),
        )

        # Encoder Path
        self.enc1 = ConvNeXtBlock3D(
            K,
            K * 2,
            kernel_size=7,
            stride=2,
            drop_path_rate=drop_rate * 0.2,
        )

        self.enc2 = nn.Sequential(
            ConvNeXtBlock3D(
                K * 2,
                K * 4,
                kernel_size=7,
                stride=2,
                drop_path_rate=drop_rate * 0.4,
            ),
            Mamba3DBlock(
                dim=K * 4,
                drop_path_rate=drop_rate * 0.4,
            ),
        )

        self.enc3 = nn.Sequential(
            ConvNeXtBlock3D(
                K * 4,
                K * 8,
                kernel_size=7,
                stride=2,
                drop_path_rate=drop_rate * 0.6,
            ),
            Mamba3DBlock(
                dim=K * 8,
                drop_path_rate=drop_rate * 0.6,
            ),
        )

        # Bottleneck: Lite-ASPP3D + 3D Mamba
        self.bottleneck = nn.Sequential(
            LiteDepthwiseASPPModule3D(K * 8, K * 8, rates=(1, 3, 6)),
            Mamba3DBlock(
                dim=K * 8,
                drop_path_rate=bottleneck_drop_rate,
            ),
        )

        # Decoder Path
        self.dec3 = UpDecoderBlock3D(
            in_dim=K * 8,
            skip_dim=K * 4,
            out_dim=K * 4,
            use_mamba=True,
            drop_path_rate=drop_rate * 0.4,
        )
        self.dec2 = UpDecoderBlock3D(
            in_dim=K * 4,
            skip_dim=K * 2,
            out_dim=K * 2,
            use_mamba=False,
            drop_path_rate=drop_rate * 0.2,
        )
        self.dec1 = UpDecoderBlock3D(
            in_dim=K * 2,
            skip_dim=K,
            out_dim=K,
            use_mamba=False,
            drop_path_rate=0.0,
        )

        # Final Classifier
        self.final = nn.Conv3d(K, out_dim, kernel_size=1)

        self.apply(model_weights_init_3d)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing dynamically."""
        self.use_checkpoint = enable

    def _ckpt(self, module: nn.Module, *args) -> Tensor:
        """Helper function to execute standard forward or checkpointed forward with re-entrant mode."""
        if self.use_checkpoint and self.training:
            if not any(isinstance(a, Tensor) and a.requires_grad for a in args):
                for a in args:
                    if isinstance(a, Tensor):
                        a.requires_grad_(True)
                        break
            return checkpoint(module, *args, use_reentrant=True)
        return module(*args)

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        x0 = self._ckpt(self.stem, input)
        x1 = self._ckpt(self.enc1, x0)
        x2 = self._ckpt(self.enc2, x1)
        x3 = self._ckpt(self.enc3, x2)

        b = self._ckpt(self.bottleneck, x3)

        d3 = self._ckpt(self.dec3, b, x2)
        d2 = self._ckpt(self.dec2, d3, x1)
        d1 = self._ckpt(self.dec1, d2, x0)

        out = self.final(d1)
        return out

    def init_weights(self):
        self.apply(model_weights_init_3d)
