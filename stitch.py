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
    crop_z_margins: bool = False,
) -> np.ndarray:
    """
    Organ-specific 3D post-processing tailored for axial CT volumes.
    Iterates sequentially through class indices 1 to 4.
    Input shape expected: (H, W, Z) where Z is the axial slice index.
    
    Classes:
        1: Esophagus
        2: Heart
        3: Trachea
        4: Aorta
    """
    cleaned_arr = np.zeros_like(arr)
    struct_3d_26 = generate_binary_structure(3, 3)
    struct_2d_8 = generate_binary_structure(2, 2)

    H, W, Z = arr.shape
    z_min, z_max = 0, Z
    if crop_z_margins:
        z_min = int(Z * 0.05)
        z_max = int(Z * 0.95)

    # Process sequentially by class index (1: Esophagus, 2: Heart, 3: Trachea, 4: Aorta)
    for c in range(1, num_classes):
        mask = arr[:, :, z_min:z_max] == c
        if not mask.any():
            continue

        # -------------------------------------------------------------
        # Class 1: Esophagus (Thin vertical structure across axial slices)
        # -------------------------------------------------------------
        if c == 1:
            labeled_mask, num_features = label(mask, structure=struct_3d_26)
            if num_features > 0:
                sizes = np.bincount(labeled_mask.ravel())
                sizes[0] = 0

                # Filter small noise artifacts while keeping largest valid components
                valid_labels = np.where(sizes >= 250)[0]
                if len(valid_labels) > 0:
                    mask = np.isin(labeled_mask, valid_labels)
                else:
                    mask = labeled_mask == np.argmax(sizes)

            # Bridge axial Z-axis gaps (up to 2-3 missing slices vertically)
            z_gap_kernel = np.zeros((1, 1, 5), dtype=bool)
            z_gap_kernel[0, 0, :] = True
            mask = binary_closing(mask, structure=z_gap_kernel)

        # -------------------------------------------------------------
        # Class 2: Heart (Large compact structure)
        # -------------------------------------------------------------
        elif c == 2:
            # 2D hole filling on each axial slice
            for z in range(mask.shape[2]):
                if mask[:, :, z].any():
                    mask[:, :, z] = binary_fill_holes(mask[:, :, z])

            # 3D closing to smooth volumetric boundaries
            mask = binary_closing(mask, structure=struct_3d_26, iterations=2)

            # Keep only the single largest contiguous component
            labeled_mask, num_features = label(mask, structure=struct_3d_26)
            if num_features > 0:
                sizes = np.bincount(labeled_mask.ravel())
                sizes[0] = 0
                mask = labeled_mask == np.argmax(sizes)

        # -------------------------------------------------------------
        # Class 3: Trachea (Continuous central airway)
        # -------------------------------------------------------------
        elif c == 3:
            # 2D slice-wise hole filling (airway lumen)
            for z in range(mask.shape[2]):
                if mask[:, :, z].any():
                    mask[:, :, z] = binary_fill_holes(mask[:, :, z])

            labeled_mask, num_features = label(mask, structure=struct_3d_26)
            if num_features > 0:
                sizes = np.bincount(labeled_mask.ravel())
                sizes[0] = 0
                # Retain the largest component to prevent empty predictions (prevents inf HD95)
                mask = labeled_mask == np.argmax(sizes)

            # Close minor Z-axis gaps between slices
            z_kernel = np.zeros((1, 1, 3), dtype=bool)
            z_kernel[0, 0, :] = True
            mask = binary_closing(mask, structure=z_kernel)

        # -------------------------------------------------------------
        # Class 4: Aorta (Ascending/Descending tubular sections)
        # -------------------------------------------------------------
        elif c == 4:
            # Enforce solid cross-sections per axial slice
            for z in range(mask.shape[2]):
                if mask[:, :, z].any():
                    slice_2d = binary_closing(mask[:, :, z], structure=struct_2d_8, iterations=1)
                    mask[:, :, z] = binary_fill_holes(slice_2d)

            labeled_mask, num_features = label(mask, structure=struct_3d_26)
            if num_features > 0:
                sizes = np.bincount(labeled_mask.ravel())
                sizes[0] = 0

                # Retain primary components (> 1500 voxels) to keep both arch & descending sections
                valid_labels = np.where(sizes >= 1500)[0]
                if len(valid_labels) > 0:
                    mask = np.isin(labeled_mask, valid_labels)
                else:
                    mask = labeled_mask == np.argmax(sizes)

        # Prevent overwriting previously assigned voxel space
        empty_space = cleaned_arr[:, :, z_min:z_max] == 0
        cleaned_arr[:, :, z_min:z_max][mask & empty_space] = c

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

    # Scale normalization back to standard class integers
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
