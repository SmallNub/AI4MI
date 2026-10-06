#!/usr/bin/env python3

from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# =====================================================================
# Space-To-Depth & Depth-To-Space Modules (Channels-Last-3D Compatible)
# =====================================================================


class SpaceToDepth3D(nn.Module):
    """Folds spatial voxels into channels with zero information loss.

    Preserves `torch.channels_last_3d` memory format.
    Input:  [B, C, D, H, W]
    Output: [B, C * (bd * bh * bw), D // bd, H // bh, W // bw]
    """

    def __init__(self, block_size: Union[int, Tuple[int, int, int]] = (1, 2, 2)):
        super().__init__()
        self.bs = (
            block_size
            if isinstance(block_size, tuple)
            else (block_size, block_size, block_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, d, h, w = x.shape
        bd, bh, bw = self.bs

        assert (
            d % bd == 0 and h % bh == 0 and w % bw == 0
        ), f"Spatial dimensions ({d},{h},{w}) must be divisible by block_size={self.bs}"

        x = x.view(b, c, d // bd, bd, h // bh, bh, w // bw, bw)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
        out = x.reshape(b, c * (bd * bh * bw), d // bd, h // bh, w // bw)
        return out.contiguous(memory_format=torch.channels_last_3d)


class DepthToSpace3D(nn.Module):
    """Exact spatial reconstruction inverse of SpaceToDepth3D.

    Input:  [B, C, D, H, W]
    Output: [B, C // (bd * bh * bw), D * bd, H * bh, W * bw]
    """

    def __init__(self, block_size: Union[int, Tuple[int, int, int]] = (1, 2, 2)):
        super().__init__()
        self.bs = (
            block_size
            if isinstance(block_size, tuple)
            else (block_size, block_size, block_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, d, h, w = x.shape
        bd, bh, bw = self.bs
        out_c = c // (bd * bh * bw)

        x = x.view(b, out_c, bd, bh, bw, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        out = x.reshape(b, out_c, d * bd, h * bh, w * bw)
        return out.contiguous(memory_format=torch.channels_last_3d)


# =====================================================================
# Regularization, Normalization & Inits
# =====================================================================


def get_group_norm_3d(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Fast, memory-efficient GroupNorm using optimized native CUDA kernels."""
    for g in [max_groups, 4, 2]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """Stochastic Depth / DropPath for 5D volumetric tensors."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath3D(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


def model_weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d, nn.LayerNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class PatchStableGRN3D(nn.Module):
    """Patch-invariant Global Response Normalization for 3D tensors.

    Calibrates channel features without spatial aggregation over patch boundaries,
    preventing seam artifacts during sliding window inference.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Channel-wise L2 norm per voxel location
        nx = x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


# =====================================================================
# State-of-the-Art 3D Blocks
# =====================================================================


class FastSqueezeExcite3D(nn.Module):
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


class FactorizedConvNeXtV2Block3D(nn.Module):
    """3D ConvNeXt-V2 block featuring Patch-Stable GRN and anisotropic factorized depthwise convs."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 7, 7),
        stride: Union[int, Tuple[int, int, int]] = 1,
        expand_ratio: float = 1.5,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        ks = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        st = stride if isinstance(stride, tuple) else (stride, stride, stride)

        pd = (ks[0] // 2, 0, 0)
        ps = (0, ks[1] // 2, ks[2] // 2)

        stride_s = (1, st[1], st[2])
        stride_d = (st[0], 1, 1)

        hidden_dim = int(in_dim * expand_ratio)

        self.dwconv_spatial = nn.Conv3d(
            in_dim,
            in_dim,
            kernel_size=(1, ks[1], ks[2]),
            stride=stride_s,
            padding=ps,
            groups=in_dim,
            bias=False,
        )
        self.dwconv_depth = nn.Conv3d(
            in_dim,
            in_dim,
            kernel_size=(ks[0], 1, 1),
            stride=stride_d,
            padding=pd,
            groups=in_dim,
            bias=False,
        )

        self.norm = get_group_norm_3d(in_dim)
        self.pwconv1 = nn.Conv3d(in_dim, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.grn = PatchStableGRN3D(hidden_dim)
        self.pwconv2 = nn.Conv3d(hidden_dim, out_dim, kernel_size=1, bias=False)
        self.se = FastSqueezeExcite3D(out_dim)

        self.drop_path = (
            DropPath3D(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        if st != (1, 1, 1) or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=st, bias=False),
                get_group_norm_3d(out_dim),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        res = self.shortcut(x)
        x = self.dwconv_spatial(x)
        x = self.dwconv_depth(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.se(x)
        return res + self.drop_path(x)


class EfficientUpDecoderBlock3D(nn.Module):
    """Ultra-low-memory Decoder Block using channel-gated skip fusion."""

    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        out_dim: int,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.proj_skip = (
            nn.Conv3d(skip_dim, in_dim, kernel_size=1, bias=False)
            if skip_dim != in_dim
            else nn.Identity()
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_dim, in_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.reduce = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=1, bias=False),
            get_group_norm_3d(out_dim),
            nn.SiLU(inplace=True),
        )
        self.spatial = FactorizedConvNeXtV2Block3D(
            out_dim, out_dim, kernel_size=(3, 5, 5), drop_path_rate=drop_path_rate
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x_up = F.interpolate(
            x, size=skip.shape[2:], mode="trilinear", align_corners=False
        )
        skip_proj = self.proj_skip(skip)
        fused = x_up + (skip_proj * self.gate(x_up))
        return self.spatial(self.reduce(fused))


# =====================================================================
# Main Architecture: ImprovedENet3D (Patch & Sliding Window Optimized)
# =====================================================================


class ImprovedENet3D(nn.Module):
    """Patch-optimized 3D UNet for SEGTHOR segmentation (32x128x128 input patches).

    Features SpaceToDepth Stem, Anisotropic Downsampling, Patch-Stable GRN,
    Anisotropic Bottleneck Dilations, and optional Deep Supervision.
    """

    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 5,  # SEGTHOR: 0=BG, 1=Esophagus, 2=Heart, 3=Trachea, 4=Aorta
        drop_rate: float = 0.1,
        bottleneck_drop_rate: float = 0.2,
        stem_block_size: Tuple[int, int, int] = (1, 2, 2),
        deep_supervision: bool = False,
        **kwargs,
    ):
        super().__init__()
        K: int = kwargs.get("kernels", 16)
        self.stem_block_size = stem_block_size
        self.deep_supervision = deep_supervision

        # SpaceToDepth Stem (Input 32x128x128 -> Stem Output 32x64x64)
        bd, bh, bw = self.stem_block_size
        s2d_channels = in_dim * (bd * bh * bw)

        self.stem_s2d = SpaceToDepth3D(block_size=self.stem_block_size)
        self.stem_proj = nn.Conv3d(s2d_channels, K, kernel_size=1, bias=False)
        self.stem_norm = get_group_norm_3d(K)
        self.stem_act = nn.SiLU(inplace=True)
        self.stem_block = FactorizedConvNeXtV2Block3D(
            K, K, kernel_size=(3, 3, 3), drop_path_rate=0.0
        )

        # Encoders (Anisotropic strides to preserve depth resolution)
        # enc1: (32,64,64) -> (32,32,32)
        self.enc1 = FactorizedConvNeXtV2Block3D(
            K,
            K * 2,
            stride=(1, 2, 2),
            kernel_size=(3, 5, 5),
            drop_path_rate=drop_rate * 0.2,
        )
        # enc2: (32,32,32) -> (16,16,16)
        self.enc2 = nn.Sequential(
            FactorizedConvNeXtV2Block3D(
                K * 2,
                K * 4,
                stride=(2, 2, 2),
                kernel_size=(3, 5, 5),
                drop_path_rate=drop_rate * 0.4,
            ),
            FactorizedConvNeXtV2Block3D(
                K * 4,
                K * 4,
                stride=1,
                kernel_size=(3, 5, 5),
                drop_path_rate=drop_rate * 0.4,
            ),
        )
        # enc3: (16,16,16) -> (8,8,8)
        self.enc3 = nn.Sequential(
            FactorizedConvNeXtV2Block3D(
                K * 4,
                K * 8,
                stride=(2, 2, 2),
                kernel_size=(3, 5, 5),
                drop_path_rate=drop_rate * 0.6,
            ),
            FactorizedConvNeXtV2Block3D(
                K * 8,
                K * 8,
                stride=1,
                kernel_size=(3, 5, 5),
                drop_path_rate=drop_rate * 0.6,
            ),
        )

        # Bottleneck: Anisotropic Dilated Depthwise Cascade (8x8x8 volume)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(
                K * 8,
                K * 8,
                kernel_size=3,
                padding=(1, 1, 1),
                dilation=(1, 1, 1),
                groups=K * 8,
                bias=False,
            ),
            nn.Conv3d(
                K * 8,
                K * 8,
                kernel_size=3,
                padding=(1, 2, 2),
                dilation=(1, 2, 2),
                groups=K * 8,
                bias=False,
            ),
            nn.Conv3d(
                K * 8,
                K * 8,
                kernel_size=3,
                padding=(2, 4, 4),
                dilation=(2, 4, 4),
                groups=K * 8,
                bias=False,
            ),
            get_group_norm_3d(K * 8),
            nn.SiLU(inplace=True),
            FactorizedConvNeXtV2Block3D(
                K * 8, K * 8, kernel_size=(3, 5, 5), drop_path_rate=bottleneck_drop_rate
            ),
        )

        # Decoders
        self.dec3 = EfficientUpDecoderBlock3D(
            in_dim=K * 8, skip_dim=K * 4, out_dim=K * 4, drop_path_rate=drop_rate * 0.4
        )
        self.dec2 = EfficientUpDecoderBlock3D(
            in_dim=K * 4, skip_dim=K * 2, out_dim=K * 2, drop_path_rate=drop_rate * 0.2
        )
        self.dec1 = EfficientUpDecoderBlock3D(
            in_dim=K * 2, skip_dim=K, out_dim=K, drop_path_rate=0.0
        )

        # Final Head with DepthToSpace pixel unshuffle (Restores full 32x128x128 resolution)
        d2s_out_channels = out_dim * (bd * bh * bw)
        self.final_conv = nn.Sequential(
            nn.Conv3d(K, d2s_out_channels, kernel_size=1, bias=False),
            get_group_norm_3d(d2s_out_channels),
            nn.SiLU(inplace=True),
        )
        self.d2s = DepthToSpace3D(block_size=self.stem_block_size)

        # Optional Deep Supervision Heads
        if self.deep_supervision:
            self.ds_dec2 = nn.Conv3d(K * 2, out_dim, kernel_size=1)
            self.ds_dec3 = nn.Conv3d(K * 4, out_dim, kernel_size=1)

        self.apply(model_weights_init)

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        # Stem SpaceToDepth: (B, 1, 32, 128, 128) -> (B, K, 32, 64, 64)
        x0 = self.stem_s2d(input)
        x0 = self.stem_proj(x0)
        x0 = self.stem_norm(x0)
        x0 = self.stem_block(self.stem_act(x0))

        # Encoders
        x1 = self.enc1(x0)  # (B, K*2, 32, 32, 32)
        x2 = self.enc2(x1)  # (B, K*4, 16, 16, 16)
        x3 = self.enc3(x2)  # (B, K*8, 8, 8, 8)

        # Bottleneck
        b = self.bottleneck(x3)  # (B, K*8, 8, 8, 8)

        # Decoders
        d3 = self.dec3(b, x2)  # (B, K*4, 16, 16, 16)
        d2 = self.dec2(d3, x1)  # (B, K*2, 32, 32, 32)
        d1 = self.dec1(d2, x0)  # (B, K, 32, 64, 64)

        # Final projection and DepthToSpace back to (B, out_dim, 32, 128, 128)
        out = self.final_conv(d1)
        logits = self.d2s(out)

        if self.training and self.deep_supervision:
            ds2 = F.interpolate(
                self.ds_dec2(d2),
                size=input.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            ds3 = F.interpolate(
                self.ds_dec3(d3),
                size=input.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            return logits, [ds2, ds3]

        return logits

    def init_weights(self):
        self.apply(model_weights_init)
