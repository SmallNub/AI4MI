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
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask
import torchvision.transforms.v2 as v2


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


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ["train", "val", "test"]

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / "img"
    full_path = root / subset / "gt"

    images: list[Path] = sorted(img_path.glob("*.npy"))
    full_labels: list[Path | None]
    if subset != "test":
        full_labels = sorted(full_path.glob("*.npy"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(
        self,
        subset,
        root_dir,
        img_transform=None,
        gt_transform=None,
        augment=False,
        equalize=False,
        debug=False,
        z_window=1,
        drop_empty=False,
    ):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize
        self.test_mode: bool = subset == "test"
        self.z_window: int = z_window
        self.half_z: int = z_window // 2
        self.drop_empty: bool = drop_empty

        self.full_files = make_dataset(root_dir, subset)

        if self.drop_empty and not self.test_mode:
            print(
                f">> Filtering empty slices (retaining only labels 1, 2, 3, 4) for {subset}..."
            )
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

        if self.augmentation and subset == "train":
            self.spatial_transform = v2.Compose(
                [
                    v2.RandomAffine(
                        degrees=(-5, 5),
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05),
                        interpolation=v2.InterpolationMode.BILINEAR,
                    )
                ]
            )
        else:
            self.spatial_transform = None

        print(
            f">> Created {subset} dataset with {len(self)} images "
            f"(Augmentation: {self.spatial_transform is not None}, Z={self.z_window}, Drop Empty: {self.drop_empty})..."
        )

    def _get_valid_index(self, center_full_idx: int, offset: int) -> int:
        target_idx = center_full_idx + offset

        if target_idx < 0 or target_idx >= len(self.full_files):
            return center_full_idx

        center_stem = self.full_files[center_full_idx][0].stem
        target_stem = self.full_files[target_idx][0].stem

        center_patient = center_stem.rsplit("_", 1)[0]
        target_patient = target_stem.rsplit("_", 1)[0]

        if center_patient != target_patient:
            return center_full_idx

        return target_idx

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index) -> dict:
        img_tensors = []

        for offset in range(-self.half_z, self.half_z + 1):
            valid_idx = self._get_valid_index(index, offset)
            img_path, _ = self.files[valid_idx]
            img_tensors.append(self.img_transform(np.load(img_path)))

        if self.z_window == 1:
            stacked_img = img_tensors[0]
        else:
            stacked_img = torch.stack(img_tensors, dim=0)

        center_stem = self.files[index][0].stem
        data_dict = {"images": stacked_img, "stems": center_stem}

        if not self.test_mode:
            _, gt_path = self.files[index]
            gt = self.gt_transform(np.load(gt_path))

            if self.spatial_transform is not None:
                if self.z_window > 1:
                    Z, C, H, W = stacked_img.shape
                    flat_img = stacked_img.view(Z * C, H, W)
                    flat_img, gt = self.spatial_transform(flat_img, Mask(gt))
                    stacked_img = flat_img.view(Z, C, H, W)
                else:
                    stacked_img, gt = self.spatial_transform(stacked_img, Mask(gt))

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
        self.shape = (256, 256)  # Default SEGTHOR spatial shape
        self.resample = resample
        self.target_spacing = target_spacing

        # 1. Folder routing
        raw_folder = "test" if self.test_mode else "train"
        self.data_path = self.root_dir / raw_folder

        # 2. Filter directories to include ONLY valid patient folders containing .nii.gz files
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

        # 3. Dynamic train/val split
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

        print(
            f">> Created 3D {subset} dataset with {len(self.files)} volumes "
            f"(Resampling: {self.resample}, Spacing: {self.target_spacing if self.resample else 'N/A'}, Found {len(all_ids)} total valid patient folders)..."
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        patient_id = self.files[index]
        patient_path = self.data_path / patient_id

        target_z = 128  # Standard depth

        # 1. Load CT Volume
        ct_path = patient_path / f"{patient_id}.nii.gz"
        
        if self.resample:
            ct_sitk = sitk.ReadImage(str(ct_path))
            ct_processed = resample_sitk_image(ct_sitk, target_spacing=self.target_spacing, is_mask=False)
            ct_arr_z_hw = sitk.GetArrayFromImage(ct_processed)  # [Z, H, W]
            ct = np.transpose(ct_arr_z_hw, (1, 2, 0))  # Convert to [H, W, Z] to match Nibabel convention
            
            spacing = ct_processed.GetSpacing()
            direction = ct_processed.GetDirection()
            origin = ct_processed.GetOrigin()
            affine = np.eye(4)
            affine[:3, :3] = np.array(direction).reshape(3, 3) @ np.diag(spacing)
            affine[:3, 3] = origin
            orig_shape = ct.shape
        else:
            # Original Nibabel implementation
            ct_nii = nib.load(str(ct_path))
            ct = np.asarray(ct_nii.dataobj)
            affine = ct_nii.affine
            orig_shape = ct.shape

        norm_ct = norm_arr(ct)  # [H, W, Z]
        ct_tensor = (
            torch.from_numpy(norm_ct).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        )

        # Resize to fixed target depth [1, target_z, 256, 256]
        ct_resized = torch.nn.functional.interpolate(
            ct_tensor,
            size=(target_z, self.shape[0], self.shape[1]),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)  # Shape: [1, 128, 256, 256]

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
                gt_processed = resample_sitk_image(gt_sitk, target_spacing=self.target_spacing, is_mask=True)
                gt_arr_z_hw = sitk.GetArrayFromImage(gt_processed)  # [Z, H, W]
                gt = np.transpose(gt_arr_z_hw, (1, 2, 0))  # Convert to [H, W, Z]
            else:
                # Original Nibabel implementation
                gt = np.asarray(nib.load(str(gt_path)).dataobj)  # [H, W, Z]

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
            gt_one_hot = gt_one_hot.permute(
                3, 0, 1, 2
            ).float()  # Switch to float for interpolation, back to bool later

            # --- 3D AUGMENTATIONS (Train Only) ---
            if self.augment and self.subset == "train":
                # 1. Random Scaling and Translation via Affine Grid
                if random.random() > 0.3:
                    scale = random.uniform(0.9, 1.1)
                    tx = random.uniform(-0.1, 0.1)
                    ty = random.uniform(-0.1, 0.1)
                    tz = random.uniform(-0.1, 0.1)

                    theta = torch.tensor(
                        [[scale, 0, 0, tx], [0, scale, 0, ty], [0, 0, scale, tz]],
                        dtype=torch.float32,
                    ).unsqueeze(0)

                    grid = torch.nn.functional.affine_grid(
                        theta,
                        [1, 1, target_z, self.shape[0], self.shape[1]],
                        align_corners=False,
                    )
                    ct_resized = torch.nn.functional.grid_sample(
                        ct_resized.unsqueeze(0),
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

                if random.random() > 0.5:
                    k = random.choice([1, 2, 3])
                    ct_resized = torch.rot90(ct_resized, k, dims=[2, 3])
                    gt_one_hot = torch.rot90(gt_one_hot, k, dims=[2, 3])

                if random.random() > 0.5:
                    ct_resized = torch.flip(ct_resized, dims=[2])
                    gt_one_hot = torch.flip(gt_one_hot, dims=[2])

                if random.random() > 0.5:
                    ct_resized = torch.flip(ct_resized, dims=[3])
                    gt_one_hot = torch.flip(gt_one_hot, dims=[3])

                # 3. Random Gaussian Noise (Applied ONLY to images)
                if random.random() > 0.5:
                    noise = (
                        torch.randn_like(ct_resized) * 0.1
                    )
                    ct_resized = ct_resized + noise

            # Convert ground truth back to boolean format for your loss function
            data_dict["images"] = ct_resized
            data_dict["gts"] = gt_one_hot.bool()

        return data_dict