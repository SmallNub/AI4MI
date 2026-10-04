#!/usr/bin/env python3.7

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

import argparse
import pickle
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
from skimage.transform import resize

from utils import map_, tqdm_


def norm_arr(
    ct: np.ndarray, window_center: int = 40, window_width: int = 400
) -> np.ndarray:
    casted = ct.astype(np.float32)

    # Soft Tissue / Mediastinum HU Clipping (-160 HU to +240 HU)
    min_hu = window_center - (window_width / 2.0)
    max_hu = window_center + (window_width / 2.0)
    clipped = np.clip(casted, min_hu, max_hu)

    # Z-score Standardization
    mean = clipped.mean()
    std = clipped.std() + 1e-8

    return ((clipped - mean) / std).astype(np.float32)


def sanity_ct(ct, x, y, z, dx, dy, dz, skip_preprocessing: bool = False) -> bool:
    if skip_preprocessing:
        # Preprocessed volumes are float32 Z-score normalized
        assert np.issubdtype(ct.dtype, np.floating), f"Expected float, got {ct.dtype}"
        return True

    # Assertions for Raw CT Volumes
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8, np.int16, np.int32], gt.dtype
    return True


resize_: Callable = partial(
    resize, mode="constant", preserve_range=True, anti_aliasing=False
)


def slice_patient(
    id_: str,
    dest_path: Path,
    source_path: Path,
    shape: tuple[int, int],
    test_mode: bool = False,
    skip_preprocessing: bool = False,
) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (
        (id_path / f"{id_}.nii.gz")
        if not test_mode
        else (source_path / "test" / f"{id_}.nii.gz")
    )

    # Handle alternate raw test path structures
    if not ct_path.exists() and test_mode:
        ct_path = source_path / "test" / id_ / f"{id_}.nii.gz"

    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()[:3]

    assert sanity_ct(ct, *ct.shape, dx, dy, dz, skip_preprocessing=skip_preprocessing)

    gt: np.ndarray
    if not test_mode:
        gt_path: Path = id_path / "GT.nii.gz"
        gt_nib = nib.load(str(gt_path))
        gt = np.asarray(gt_nib.dataobj).astype(np.uint8)
        assert sanity_gt(gt, ct)
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    # Skip normalization step if already preprocessed by the NIfTI script
    to_slice_ct = ct.astype(np.float32) if skip_preprocessing else norm_arr(ct)
    to_slice_gt = gt

    target_shape = tuple(shape)

    for idz in range(z):
        # Apply 2D spatial resizing only if dimensions do not match target shape
        if to_slice_ct[:, :, idz].shape != target_shape:
            img_slice = resize_(to_slice_ct[:, :, idz], target_shape).astype(np.float32)
            gt_slice = resize_(to_slice_gt[:, :, idz], target_shape, order=0).astype(
                np.uint8
            )
        else:
            img_slice = to_slice_ct[:, :, idz].astype(np.float32)
            gt_slice = to_slice_gt[:, :, idz].astype(np.uint8)

        assert img_slice.shape == gt_slice.shape
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        assert set(np.unique(gt_slice)) <= set([0, 1, 2, 3, 4]), np.unique(gt_slice)

        arrays: list[np.ndarray] = [img_slice, gt_slice]
        subfolders: list[str] = ["img", "gt"]

        for save_subfolder, data in zip(subfolders, arrays):
            filename = f"{id_}_{idz:04d}.npy"
            save_path: Path = Path(dest_path, save_subfolder)
            save_path.mkdir(parents=True, exist_ok=True)

            np.save(str(save_path / filename), data)

    return float(dx), float(dy), float(dz)


def get_splits(
    src_path: Path, retains: int, fold: int
) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = sorted(map_(lambda p: p.name, (src_path / "train").glob("*")))
    print(f"Found {len(ids)} training/val patient IDs")
    assert len(ids) > retains

    random.shuffle(ids)
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: list[str] = ids[validation_slice]
    assert len(validation_ids) == retains

    training_ids: list[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)

    test_dir = src_path / "test"
    test_ids: list[str] = sorted(
        map_(
            lambda p: p.name if p.is_dir() else p.name.split(".")[0],
            test_dir.glob("*"),
        )
    )
    print(f"Found {len(test_ids)} test patient IDs")

    return training_ids, validation_ids, test_ids


def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)

    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    assert src_path.exists(), f"Source path {src_path} does not exist."
    assert not dest_path.exists(), f"Destination path {dest_path} already exists."

    training_ids, validation_ids, test_ids = get_splits(
        src_path, args.retains, args.fold
    )

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(
            slice_patient,
            dest_path=dest_mode,
            source_path=src_path,
            shape=tuple(args.shape),
            test_mode=(mode == "test"),
            skip_preprocessing=args.skip_preprocessing,
        )

        iterator = tqdm_(split_ids)
        if args.process == 1:
            resolutions = list(map(pfun, iterator))
        else:
            num_workers = None if args.process == -1 else args.process
            with Pool(processes=num_workers) as pool:
                resolutions = pool.map(pfun, iterator)

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    dest_path.mkdir(parents=True, exist_ok=True)
    with open(dest_path / "spacing.pkl", "wb") as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionary to {dest_path / 'spacing.pkl'}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D Slicing script for NIfTI volumes")
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Input directory containing NIfTI files",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        required=True,
        help="Output directory for sliced 2D .npy files",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Target 2D slice height/width",
    )
    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip HU windowing, Z-score normalization, and raw CT shape/spacing assertions if input NIfTI files are already preprocessed.",
    )
    parser.add_argument(
        "--retains",
        type=int,
        default=25,
        help="Number of retained patients for the validation split",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--process",
        "-p",
        type=int,
        default=1,
        help="Number of core processes to use (-1 for all cores)",
    )
    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
