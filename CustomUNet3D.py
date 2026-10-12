#!/usr/bin/env python3

from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpaceToDepth3D(nn.Module):
    """Folds spatial voxels into channels with zero information loss."""

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
    """Exact spatial reconstruction inverse of SpaceToDepth3D."""

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


def get_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
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
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d, nn.LayerNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class GRN(nn.Module):
    """Global Response Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Channel-wise L2 norm per voxel location
        nx = x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class SqueezeExcite3D(nn.Module):
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


class ConvNeXtBlock3D(nn.Module):
    """3D ConvNeXt block."""

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

        self.norm = get_group_norm(in_dim)
        self.pwconv1 = nn.Conv3d(in_dim, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.grn = GRN(hidden_dim)
        self.pwconv2 = nn.Conv3d(hidden_dim, out_dim, kernel_size=1, bias=False)
        self.se = SqueezeExcite3D(out_dim)

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        if st != (1, 1, 1) or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=st, bias=False),
                get_group_norm(out_dim),
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


class UpDecoderBlock3D(nn.Module):
    """Decoder Block using channel-gated skip fusion."""

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
            get_group_norm(out_dim),
            nn.SiLU(inplace=True),
        )
        self.spatial = ConvNeXtBlock3D(
            out_dim, out_dim, kernel_size=(3, 5, 5), drop_path_rate=drop_path_rate
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x_up = F.interpolate(
            x, size=skip.shape[2:], mode="trilinear", align_corners=False
        )
        skip_proj = self.proj_skip(skip)
        fused = x_up + (skip_proj * self.gate(x_up))
        return self.spatial(self.reduce(fused))


class CustomUNet3D(nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 5,
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

        # SpaceToDepth Stem
        bd, bh, bw = self.stem_block_size
        s2d_channels = in_dim * (bd * bh * bw)

        self.stem_s2d = SpaceToDepth3D(block_size=self.stem_block_size)
        self.stem_proj = nn.Conv3d(s2d_channels, K, kernel_size=1, bias=False)
        self.stem_norm = get_group_norm(K)
        self.stem_act = nn.SiLU(inplace=True)
        self.stem_block = ConvNeXtBlock3D(
            K, K, kernel_size=(3, 3, 3), drop_path_rate=0.0
        )

        # Encoders
        self.enc1 = ConvNeXtBlock3D(
            K,
            K * 2,
            stride=(1, 2, 2),
            kernel_size=(3, 5, 5),
            drop_path_rate=drop_rate * 0.2,
        )
        self.enc2 = nn.Sequential(
            ConvNeXtBlock3D(
                K * 2,
                K * 4,
                stride=(2, 2, 2),
                kernel_size=(3, 5, 5),
                drop_path_rate=drop_rate * 0.4,
            ),
            ConvNeXtBlock3D(
                K * 4,
                K * 4,
                stride=1,
                kernel_size=(3, 5, 5),
                drop_path_rate=drop_rate * 0.4,
            ),
        )
        self.enc3 = nn.Sequential(
            ConvNeXtBlock3D(
                K * 4,
                K * 8,
                stride=(2, 2, 2),
                kernel_size=(3, 5, 5),
                drop_path_rate=drop_rate * 0.6,
            ),
            ConvNeXtBlock3D(
                K * 8,
                K * 8,
                stride=1,
                kernel_size=(3, 5, 5),
                drop_path_rate=drop_rate * 0.6,
            ),
        )

        # Bottleneck
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
            get_group_norm(K * 8),
            nn.SiLU(inplace=True),
            ConvNeXtBlock3D(
                K * 8, K * 8, kernel_size=(3, 5, 5), drop_path_rate=bottleneck_drop_rate
            ),
        )

        # Decoders
        self.dec3 = UpDecoderBlock3D(
            in_dim=K * 8, skip_dim=K * 4, out_dim=K * 4, drop_path_rate=drop_rate * 0.4
        )
        self.dec2 = UpDecoderBlock3D(
            in_dim=K * 4, skip_dim=K * 2, out_dim=K * 2, drop_path_rate=drop_rate * 0.2
        )
        self.dec1 = UpDecoderBlock3D(
            in_dim=K * 2, skip_dim=K, out_dim=K, drop_path_rate=0.0
        )

        # Final Head
        d2s_out_channels = out_dim * (bd * bh * bw)
        self.final_conv = nn.Sequential(
            nn.Conv3d(K, d2s_out_channels, kernel_size=1, bias=False),
            get_group_norm(d2s_out_channels),
            nn.SiLU(inplace=True),
        )
        self.d2s = DepthToSpace3D(block_size=self.stem_block_size)

        self.apply(model_weights_init)

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        # Stem
        x0 = self.stem_s2d(input)
        x0 = self.stem_proj(x0)
        x0 = self.stem_norm(x0)
        x0 = self.stem_block(self.stem_act(x0))

        # Encoders
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Bottleneck
        b = self.bottleneck(x3)

        # Decoders
        d3 = self.dec3(b, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        # Final projection
        out = self.final_conv(d1)
        logits = self.d2s(out)

        return logits

    def init_weights(self):
        self.apply(model_weights_init)
