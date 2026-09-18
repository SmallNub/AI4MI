#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2024 Hoel Kervadec

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

import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Pattern

import numpy as np
import nibabel as nib
from skimage.io import imread
from skimage.transform import resize
from scipy.ndimage import (
    label,
    binary_closing,
    binary_fill_holes,
    generate_binary_structure,
)

from utils import tqdm_


def get_z(image: Path) -> int:
    return int(image.stem.split("_")[-1])


def post_process_3d(
    arr: np.ndarray,
    num_classes: int = 5,
    min_voxels: dict[int, int] | None = None,
    crop_z_margins: bool = False,
    fill_holes: bool = True,
) -> np.ndarray:
    cleaned_arr = np.zeros_like(arr)
    struct_26 = generate_binary_structure(3, 3)

    if min_voxels is None:
        min_voxels = {
            1: 2500,
            2: 40000,
            3: 1800,
            4: 8000,
        }
    Z = arr.shape[2]
    z_min, z_max = 0, Z
    if crop_z_margins:
        z_min = int(Z * 0.05)
        z_max = int(Z * 0.95)

    for c in range(1, num_classes):
        binary_mask = arr[:, :, z_min:z_max] == c
        if not binary_mask.any():
            continue

        # Close small gaps
        binary_mask = binary_closing(binary_mask, structure=struct_26, iterations=1)

        # Fill holes inside the predicted organ volumes
        if fill_holes:
            binary_mask = binary_fill_holes(binary_mask)

        # Connected Components
        labeled_mask, num_features = label(binary_mask, structure=struct_26)
        if num_features == 0:
            continue

        # Count sizes, but zero out the background (index 0) so direct indexing works safely
        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0

        # Protect against overlapping boundaries caused by morphological closing
        empty_space_mask = cleaned_arr[:, :, z_min:z_max] == 0

        if c == 1:
            # Esophagus: Filter by voxel threshold
            min_size = min_voxels.get(c, 0)
            valid_indices = np.where(component_sizes >= min_size)[0]
            valid_mask = np.isin(labeled_mask, valid_indices)

            cleaned_arr[:, :, z_min:z_max][valid_mask & empty_space_mask] = c
        else:
            # Heart, Trachea, Aorta: Retain strictly the largest contiguous component
            largest_idx = np.argmax(component_sizes)
            if component_sizes[largest_idx] >= min_voxels.get(c, 0):
                largest_mask = labeled_mask == largest_idx
                cleaned_arr[:, :, z_min:z_max][largest_mask & empty_space_mask] = c

    return cleaned_arr


def merge_patient(
    id_: str,
    dest_folder: str,
    images: list[Path],
    idxes: list[int],
    K: int,
    source_pattern: str,
    post_process: bool = True,
) -> None:
    orig_nib = nib.load(source_pattern.format(id_=id_))
    orig_shape = np.asarray(orig_nib.dataobj).shape

    X, Y, Z = orig_shape
    assert Z == len(
        idxes
    ), f"Slice count mismatch for patient {id_}: scan Z={Z}, images={len(idxes)}"

    res_arr: np.ndarray = np.zeros((X, Y, Z), dtype=np.int16)

    for idx in idxes:
        img: Path = images[idx]
        z = get_z(img)
        img_arr = imread(img)

        assert img_arr.dtype == np.uint8
        assert set(np.unique(img_arr)) <= set(range(K))

        # Nearest neighbor interpolation for segmentation masks
        resized: np.ndarray = resize(
            img_arr,
            (X, Y),
            mode="constant",
            preserve_range=True,
            anti_aliasing=False,
            order=0,
        )

        res_arr[:, :, z] = resized[...]

    assert set(np.unique(res_arr)) <= set(range(K))
    assert orig_shape == res_arr.shape, (orig_shape, res_arr.shape)

    # Scale normalization back to standard class integers (e.g., SegTHOR values)
    res_arr //= 63
    assert set(np.unique(res_arr)).issubset(
        set(range(5))
    ), f"Found unexpected class values: {np.unique(res_arr)}"

    # Apply enhanced 3D post-processing
    if post_process:
        res_arr = post_process_3d(res_arr, num_classes=5)

    new_nib = nib.nifti1.Nifti1Image(
        res_arr, affine=orig_nib.affine, header=orig_nib.header
    )
    nib.save(new_nib, (Path(dest_folder) / id_).with_suffix(".nii.gz"))


def main(args) -> None:
    images: list[Path] = list(Path(args.data_folder).glob("*.png"))
    grouping_regex: Pattern = re.compile(args.grp_regex)

    idx_map: dict[str, list[int]] = defaultdict(list)

    for i, img_path in enumerate(images):
        match = grouping_regex.match(img_path.stem)
        if match:
            patient = match.group(1)
            idx_map[patient].append(i)

    unique_patients = list(idx_map.keys())

    print(unique_patients)
    assert len(unique_patients) < len(images)
    print(
        f"Found {len(unique_patients)} unique patients out of {len(images)} images ; regex: {args.grp_regex}"
    )
    assert sum(len(idx) for idx in idx_map.values()) == len(images)

    args.dest_folder.mkdir(parents=True, exist_ok=True)

    for p in tqdm_(unique_patients):
        merge_patient(
            p,
            args.dest_folder,
            images,
            idx_map[p],
            args.num_classes,
            args.source_scan_pattern,
            post_process=args.post,
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merging slices parameters")
    parser.add_argument(
        "--data_folder",
        type=Path,
        required=True,
        help="The folder containing the images to predict",
    )
    parser.add_argument(
        "--source_scan_pattern",
        type=str,
        required=True,
        help="Pattern to get original scan to map metadata",
    )
    parser.add_argument("--dest_folder", type=Path, required=True)
    parser.add_argument("--grp_regex", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument(
        "--post",
        action="store_true",
        default=False,
        help="Enable 3D connected component post-processing",
    )

    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    main(get_args())
