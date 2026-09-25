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

from pathlib import Path
from typing import Callable
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask
import torchvision.transforms.v2 as v2


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ["train", "val", "test"]

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / "img"
    full_path = root / subset / "gt"

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != "test":
        full_labels = sorted(full_path.glob("*.png"))
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
            print(f">> Filtering empty slices (retaining only labels 1, 2, 3, 4) for {subset}...")
            self.valid_indices = []
            for idx, (_, gt_path) in enumerate(self.full_files):
                if gt_path is not None:
                    gt_arr = np.array(Image.open(gt_path))
                    # Check if slice contains any foreground target organ (labels 1, 2, 3, 4)
                    # Works for both raw classes [1, 2, 3, 4] and scaled values [63, 126, 189, 252]
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
        """Prevents indexing out of bounds against the complete file list"""
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
        # Map dataset index to the actual position in full_files
        full_idx = self.valid_indices[index]
        img_tensors = []

        # Retrieve neighbor slices from the full sequential file list
        for offset in range(-self.half_z, self.half_z + 1):
            valid_full_idx = self._get_valid_index(full_idx, offset)
            img_path, _ = self.full_files[valid_full_idx]
            img_tensors.append(self.img_transform(Image.open(img_path)))

        if self.z_window == 1:
            stacked_img = img_tensors[0]
        else:
            stacked_img = torch.stack(img_tensors, dim=0)

        center_stem = self.full_files[full_idx][0].stem
        data_dict = {"images": stacked_img, "stems": center_stem}

        if not self.test_mode:
            _, gt_path = self.full_files[full_idx]
            gt = self.gt_transform(Image.open(gt_path))

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
