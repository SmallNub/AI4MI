#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import nibabel as nib
import SimpleITK as sitk
from PIL import Image
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask
import torchvision.transforms.v2 as v2

# =============================================================================
# Helper Utilities & Preprocessing
# =============================================================================


def norm_arr(
    ct: np.ndarray, window_center: int = 40, window_width: int = 400
) -> np.ndarray:
    """Clips CT to Soft Tissue window and applies Z-score standardization."""
    casted = ct.astype(np.float32)
    min_hu = window_center - (window_width / 2.0)
    max_hu = window_center + (window_width / 2.0)
    clipped = np.clip(casted, min_hu, max_hu)
    mean = clipped.mean()
    std = clipped.std() + 1e-8
    standardized = (clipped - mean) / std
    return standardized


def resample_sitk_image(
    image: sitk.Image,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_mask: bool = False,
) -> sitk.Image:
    """Resamples a SimpleITK image to a target physical spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    if np.allclose(original_spacing, target_spacing):
        return image

    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())

    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(image)


def make_dataset(root: str | Path, subset: str) -> list[tuple[Path, Path | None]]:
    assert subset in ["train", "val", "test"]

    root = Path(root)
    img_path = root / subset / "img"
    full_path = root / subset / "gt"

    images: list[Path] = sorted(img_path.glob("*.npy"))
    full_labels: list[Path | None]
    if subset != "test":
        full_labels = sorted(full_path.glob("*.npy"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


# =============================================================================
# Augmentation Pipelines
# =============================================================================


class Segthor2DAugment:
    """
    Conservative 2D augmentation pipeline for SEGTHOR thoracic CT slices.
    Preserves strict left/right anatomical asymmetry and thin boundary precision.
    """

    def __init__(self, p_spatial: float = 0.3, p_intensity: float = 0.3):
        self.p_spatial = p_spatial
        self.p_intensity = p_intensity

        self.spatial_transform = v2.Compose(
            [
                v2.RandomAffine(
                    degrees=(-5, 5),  # Small rotation range
                    translate=(0.05, 0.05),  # Max 3% spatial shift
                    scale=(0.95, 1.05),  # Max 5% scale variation
                    interpolation=v2.InterpolationMode.BILINEAR,
                )
            ]
        )

    def __call__(self, img: torch.Tensor, mask: Mask) -> tuple[torch.Tensor, Mask]:
        # Joint Spatial Transform (Image + Mask)
        if random.random() < self.p_spatial:
            img, mask = self.spatial_transform(img, mask)

        # Intensity Transforms (Image Only)
        if random.random() < self.p_intensity:
            # Low-amplitude Gaussian Noise
            if random.random() < 0.5:
                noise_std = random.uniform(0.005, 0.02)
                img = img + torch.randn_like(img) * noise_std

            # Mild Gamma Adjustment (Non-linear contrast variation)
            if random.random() < 0.5:
                gamma = random.uniform(0.9, 1.1)
                img_min = img.min()
                img = torch.pow(torch.clamp(img - img_min, min=0.0), gamma) + img_min

        return img, mask


class Segthor3DAugment:
    """
    Conservative 3D volumetric augmentation pipeline for SEGTHOR CT volumes.
    Uses 3D affine grids and intensity adjustments while preventing anatomical flips.
    """

    def __init__(self, p_spatial: float = 0.3, p_intensity: float = 0.3):
        self.p_spatial = p_spatial
        self.p_intensity = p_intensity

    def __call__(
        self, ct_tensor: torch.Tensor, gt_one_hot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Tensors expect shape [1, Z, H, W] or [C, Z, H, W]
        _, z_dim, h_dim, w_dim = ct_tensor.shape

        # 1. Conservative 3D Spatial Scaling & Translation
        if random.random() < self.p_spatial:
            scale = random.uniform(0.95, 1.05)
            tx = random.uniform(-0.05, 0.05)
            ty = random.uniform(-0.05, 0.05)
            tz = random.uniform(-0.05, 0.05)

            theta = torch.tensor(
                [[scale, 0, 0, tx], [0, scale, 0, ty], [0, 0, scale, tz]],
                dtype=torch.float32,
            ).unsqueeze(0)

            grid = torch.nn.functional.affine_grid(
                theta,
                [1, 1, z_dim, h_dim, w_dim],
                align_corners=False,
            )

            ct_tensor = torch.nn.functional.grid_sample(
                ct_tensor.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).squeeze(0)

            gt_one_hot = torch.nn.functional.grid_sample(
                gt_one_hot.unsqueeze(0),
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(0)

        # 2. Intensity Augmentations (Applied strictly to image tensor)
        if random.random() < self.p_intensity:
            # Low-amplitude additive Gaussian noise
            if random.random() < 0.5:
                noise_std = random.uniform(0.005, 0.02)
                ct_tensor = ct_tensor + torch.randn_like(ct_tensor) * noise_std

            # Mild Volumetric Gamma correction
            if random.random() < 0.5:
                gamma = random.uniform(0.9, 1.1)
                img_min = ct_tensor.min()
                ct_tensor = (
                    torch.pow(torch.clamp(ct_tensor - img_min, min=0.0), gamma)
                    + img_min
                )

        return ct_tensor, gt_one_hot


# =============================================================================
# Datasets
# =============================================================================


class SliceDataset(Dataset):
    def __init__(
        self,
        subset: str,
        root_dir: str,
        img_transform: Callable = None,
        gt_transform: Callable = None,
        augment: bool = False,
        equalize: bool = False,
        debug: bool = False,
        z_window: int = 1,
        drop_empty: bool = False,
        resample: bool = False,
    ):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.test_mode: bool = subset == "test"
        self.z_window: int = z_window
        self.half_z: int = z_window // 2
        self.drop_empty: bool = drop_empty

        self.full_files = make_dataset(root_dir, subset)

        if self.drop_empty and not self.test_mode:
            print(f">> Filtering empty slices for {subset}...")
            self.valid_indices = []
            for idx, (_, gt_path) in enumerate(self.full_files):
                if gt_path is not None:
                    gt_arr = np.array(Image.open(gt_path))
                    has_target_organs = np.any((gt_arr > 0) & (gt_arr <= 252))
                    if has_target_organs:
                        self.valid_indices.append(idx)
        else:
            self.valid_indices = list(range(len(self.full_files)))

        if debug:
            self.valid_indices = self.valid_indices[:10]

        # Initialize conservative 2D joint augmentation
        if augment and subset == "train":
            self.joint_transform = Segthor2DAugment(p_spatial=0.3, p_intensity=0.3)
        else:
            self.joint_transform = None

        print(
            f">> Created {subset} 2D dataset with {len(self)} images "
            f"(Augmentation: {self.joint_transform is not None}, Z={self.z_window})..."
        )

    def _get_valid_index(self, center_full_idx: int, offset: int) -> int:
        target_idx = center_full_idx + offset
        if target_idx < 0 or target_idx >= len(self.full_files):
            return center_full_idx

        center_stem = self.full_files[center_full_idx][0].stem
        target_stem = self.full_files[target_idx][0].stem

        if center_stem.rsplit("_", 1)[0] != target_stem.rsplit("_", 1)[0]:
            return center_full_idx

        return target_idx

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> dict:
        center_full_idx = self.valid_indices[index]

        img_tensors = []
        for offset in range(-self.half_z, self.half_z + 1):
            valid_idx = self._get_valid_index(center_full_idx, offset)
            img_path, _ = self.full_files[valid_idx]
            img_data = np.load(img_path)
            if self.img_transform is not None:
                img_data = self.img_transform(img_data)
            img_tensors.append(img_data)

        stacked_img = (
            torch.cat(img_tensors, dim=0) if len(img_tensors) > 1 else img_tensors[0]
        )
        center_stem = self.full_files[center_full_idx][0].stem
        data_dict = {"images": stacked_img, "stems": center_stem}

        if not self.test_mode:
            _, gt_path = self.full_files[center_full_idx]
            gt_data = np.load(gt_path)
            if self.gt_transform is not None:
                gt_data = self.gt_transform(gt_data)

            gt = gt_data

            if self.joint_transform is not None:
                stacked_img, gt = self.joint_transform(stacked_img, Mask(gt))

            data_dict["images"] = stacked_img
            data_dict["gts"] = gt

        return data_dict


class Segthor3DDataset(Dataset):
    def __init__(
        self,
        subset: str,
        root_dir: str,
        img_transform: Callable = None,
        gt_transform: Callable = None,
        augment: bool = False,
        equalize: bool = False,
        drop_empty: bool = False,
        debug: bool = False,
        z_window: int = 1,
        resample: bool = False,
        target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        assert subset in ["train", "val", "test"]

        self.subset = subset
        self.root_dir = Path(root_dir)
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.augment = augment
        self.test_mode = subset == "test"
        self.shape = (256, 256)
        self.resample = resample
        self.target_spacing = target_spacing

        raw_folder = "test" if self.test_mode else "train"
        self.data_path = self.root_dir / raw_folder

        all_ids = sorted(
            [
                p.name
                for p in self.data_path.glob("*")
                if p.is_dir()
                and p.name not in ["img", "gt"]
                and (p / f"{p.name}.nii.gz").exists()
            ]
        )

        if debug:
            all_ids = all_ids[:10]

        if not self.test_mode:
            random.shuffle(all_ids)
            total_volumes = len(all_ids)
            retains = 5 if total_volumes >= 5 else max(1, total_volumes)
            fold = 0

            val_slice = slice(fold * retains, (fold + 1) * retains)
            val_ids = all_ids[val_slice]
            train_ids = [x for x in all_ids if x not in val_ids]

            self.files = train_ids if subset == "train" else val_ids
        else:
            self.files = all_ids

        # Initialize conservative 3D augmentation module
        if self.augment and self.subset == "train":
            self.augmentor = Segthor3DAugment(p_spatial=0.3, p_intensity=0.3)
        else:
            self.augmentor = None

        print(
            f">> Created 3D {subset} dataset with {len(self.files)} volumes "
            f"(Augmentation: {self.augmentor is not None}, Resampling: {self.resample})..."
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        patient_id = self.files[index]
        patient_path = self.data_path / patient_id
        target_z = 128

        # 1. Load CT Volume
        ct_path = patient_path / f"{patient_id}.nii.gz"

        if self.resample:
            ct_sitk = sitk.ReadImage(str(ct_path))
            ct_processed = resample_sitk_image(
                ct_sitk, target_spacing=self.target_spacing, is_mask=False
            )
            ct_arr_z_hw = sitk.GetArrayFromImage(ct_processed)
            ct = np.transpose(ct_arr_z_hw, (1, 2, 0))

            spacing = ct_processed.GetSpacing()
            direction = ct_processed.GetDirection()
            origin = ct_processed.GetOrigin()
            affine = np.eye(4)
            affine[:3, :3] = np.array(direction).reshape(3, 3) @ np.diag(spacing)
            affine[:3, 3] = origin
            orig_shape = ct.shape
        else:
            ct_nii = nib.load(str(ct_path))
            ct = np.asarray(ct_nii.dataobj)
            affine = ct_nii.affine
            orig_shape = ct.shape

        norm_ct = norm_arr(ct)
        ct_tensor = (
            torch.from_numpy(norm_ct).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        )

        ct_resized = torch.nn.functional.interpolate(
            ct_tensor,
            size=(target_z, self.shape[0], self.shape[1]),
            mode="trilinear",
            align_corners=False,
        ).squeeze(
            0
        )  # Shape: [1, 128, 256, 256]

        data_dict = {
            "images": ct_resized,
            "stems": patient_id,
            "affine": torch.from_numpy(affine).float(),
            "orig_shape": torch.tensor(orig_shape),
        }

        # 2. Load Ground Truth
        if not self.test_mode:
            gt_path = patient_path / "GT.nii.gz"

            if self.resample:
                gt_sitk = sitk.ReadImage(str(gt_path))
                gt_processed = resample_sitk_image(
                    gt_sitk, target_spacing=self.target_spacing, is_mask=True
                )
                gt_arr_z_hw = sitk.GetArrayFromImage(gt_processed)
                gt = np.transpose(gt_arr_z_hw, (1, 2, 0))
            else:
                gt = np.asarray(nib.load(str(gt_path)).dataobj)

            gt_tensor = (
                torch.from_numpy(gt).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            )

            gt_resized = (
                torch.nn.functional.interpolate(
                    gt_tensor,
                    size=(target_z, self.shape[0], self.shape[1]),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )

            gt_one_hot = torch.nn.functional.one_hot(gt_resized, num_classes=5)
            gt_one_hot = gt_one_hot.permute(3, 0, 1, 2).float()

            # --- 3D AUGMENTATIONS ---
            if self.augmentor is not None:
                ct_resized, gt_one_hot = self.augmentor(ct_resized, gt_one_hot)

            data_dict["images"] = ct_resized
            data_dict["gts"] = gt_one_hot.bool()

        return data_dict
