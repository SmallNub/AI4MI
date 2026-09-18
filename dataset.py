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
from typing import Callable, Union
import torch

from torch import Tensor
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
    ):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize
        self.test_mode: bool = subset == "test"
        self.z_window: int = z_window
        self.half_z: int = z_window // 2

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

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
            f">> Created {subset} dataset with {len(self)} images (Augmentation: {self.spatial_transform is not None}, Z={self.z_window})..."
        )

    def _get_valid_index(self, center_idx: int, offset: int) -> int:
        """Prevents indexing out of bounds"""
        target_idx = center_idx + offset

        if target_idx < 0 or target_idx >= len(self.files):
            return center_idx

        center_stem = self.files[center_idx][0].stem
        target_stem = self.files[target_idx][0].stem

        center_patient = center_stem.rsplit("_", 1)[0]
        target_patient = target_stem.rsplit("_", 1)[0]

        if center_patient != target_patient:
            return center_idx

        return target_idx

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict:
        img_tensors = []

        for offset in range(-self.half_z, self.half_z + 1):
            valid_idx = self._get_valid_index(index, offset)
            img_path, _ = self.files[valid_idx]
            img_tensors.append(self.img_transform(Image.open(img_path)))

        # Making sure that it works with standard Enet
        if self.z_window == 1:
            stacked_img = img_tensors[0]
        else:
            stacked_img = torch.stack(img_tensors, dim=0)

        center_stem = self.files[index][0].stem
        data_dict = {"images": stacked_img, "stems": center_stem}

        if not self.test_mode:
            _, gt_path = self.files[index]
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
