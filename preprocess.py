#!/usr/bin/env python3
"""
Minimal SEGTHOR 3D NIfTI-to-NIfTI Processor
-------------------------------------------
Expects input directory structured as:
  <input_dir>/
    <patient_id>/
      <patient_id>.nii.gz
      GT.nii.gz          (optional)

Outputs identical structure to <output_dir> with resampled, windowed,
and standardized NIfTI files.
"""

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def norm_sitk_image(
    image: sitk.Image, window_center: int = -200, window_width: int = 1600
) -> sitk.Image:
    """Clips CT image to Soft Tissue window (-160 to +240 HU) and applies Z-score standardization."""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)

    # Soft Tissue HU Clipping
    min_hu = window_center - (window_width / 2.0)
    max_hu = window_center + (window_width / 2.0)
    clipped = np.clip(arr, min_hu, max_hu)

    # Z-score Standardization
    mean = clipped.mean()
    std = clipped.std() + 1e-8
    standardized = (clipped - mean) / std

    out_img = sitk.GetImageFromArray(standardized.astype(np.float32))
    out_img.CopyInformation(image)
    return out_img


def resample_sitk_image(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.5),
    is_mask: bool = False,
) -> sitk.Image:
    """Resamples SimpleITK image to a target physical voxel spacing (dx, dy, dz)."""
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


def process_patient_folder(
    patient_dir: Path,
    input_root: Path,
    output_root: Path,
    target_spacing: Tuple[float, float, float],
) -> None:
    """Processes CT volume and optional GT mask for a single patient directory."""
    patient_id = patient_dir.name
    out_patient_dir = output_root / patient_dir.relative_to(input_root)
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process CT Image
    ct_path = patient_dir / f"{patient_id}.nii.gz"
    if ct_path.exists():
        ct_sitk = sitk.ReadImage(str(ct_path))
        ct_resampled = resample_sitk_image(
            ct_sitk, target_spacing=target_spacing, is_mask=False
        )
        ct_normalized = norm_sitk_image(ct_resampled)
        sitk.WriteImage(ct_normalized, str(out_patient_dir / f"{patient_id}.nii.gz"))

    # 2. Process Ground Truth Mask (if present)
    gt_path = patient_dir / "GT.nii.gz"
    if gt_path.exists():
        gt_sitk = sitk.ReadImage(str(gt_path))
        gt_resampled = resample_sitk_image(
            gt_sitk, target_spacing=target_spacing, is_mask=True
        )
        gt_resampled = sitk.Cast(gt_resampled, sitk.sitkUInt8)
        sitk.WriteImage(gt_resampled, str(out_patient_dir / "GT.nii.gz"))


def main():
    parser = argparse.ArgumentParser(description="Pure NIfTI Preprocessor for SEGTHOR")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing patient folders",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output destination directory"
    )
    parser.add_argument(
        "--target_spacing",
        type=float,
        nargs="+",
        default=[1.0, 1.0, 2.5],
        help="Target spacing (dx dy dz) in mm",
    )
    parser.add_argument(
        "-p",
        "--process",
        type=int,
        default=-1,
        help="Multiprocessing cores (-1 for all)",
    )

    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    target_spacing = tuple(args.target_spacing)

    # Find all subdirectories containing NIfTI files
    patient_dirs = sorted(
        [
            p
            for p in input_root.rglob("*")
            if p.is_dir() and (p / f"{p.name}.nii.gz").exists()
        ]
    )

    print(f">> Processing {len(patient_dirs)} patient folders from {input_root}...")

    pfun = partial(
        process_patient_folder,
        input_root=input_root,
        output_root=output_root,
        target_spacing=target_spacing,
    )

    if args.process == 1:
        for p_dir in tqdm(patient_dirs):
            pfun(p_dir)
    else:
        num_workers = None if args.process == -1 else args.process
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap(pfun, patient_dirs), total=len(patient_dirs)))

    print(f">> Done! Processed NIfTI files saved to {output_root}")


if __name__ == "__main__":
    main()
