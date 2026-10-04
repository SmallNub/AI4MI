#!/usr/bin/env python3
"""
Universal Post-Processing & Spatial Resampling Pipeline
-------------------------------------------------------
Handles predictions from both 2D models (slice stitching) and 3D models (3D NIfTI).
Applies organ-specific 3D morphological post-processing and resamples prediction masks
back to the exact raw CT physical reference space.
"""

import argparse
import re
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    generate_binary_structure,
    label,
)
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm


# ==============================================================================
# 1. ORGAN-SPECIFIC 3D POST-PROCESSING
# ==============================================================================
def post_process_3d(
    arr: np.ndarray,
    num_classes: int = 5,
    crop_z_margins: bool = False,
) -> np.ndarray:
    """Organ-specific 3D post-processing tailored for axial CT volumes (H, W, Z)."""
    cleaned_arr = np.zeros_like(arr, dtype=np.uint8)
    struct_3d_26 = generate_binary_structure(3, 3)
    struct_2d_8 = generate_binary_structure(2, 2)

    H, W, Z = arr.shape
    z_min, z_max = 0, Z
    if crop_z_margins:
        z_min = int(Z * 0.05)
        z_max = int(Z * 0.95)

    for c in range(1, num_classes):
        mask = arr[:, :, z_min:z_max] == c
        if not mask.any():
            continue

        # Class 1: Esophagus (Thin vertical structure)
        if c == 1:
            labeled_mask, num_features = label(mask, structure=struct_3d_26)
            if num_features > 0:
                sizes = np.bincount(labeled_mask.ravel())
                sizes[0] = 0
                valid_labels = np.where(sizes >= 250)[0]
                if len(valid_labels) > 0:
                    mask = np.isin(labeled_mask, valid_labels)
                else:
                    mask = labeled_mask == np.argmax(sizes)

            z_gap_kernel = np.zeros((1, 1, 5), dtype=bool)
            z_gap_kernel[0, 0, :] = True
            mask = binary_closing(mask, structure=z_gap_kernel)

        # Class 2: Heart (Large compact structure)
        elif c == 2:
            for z in range(mask.shape[2]):
                if mask[:, :, z].any():
                    mask[:, :, z] = binary_fill_holes(mask[:, :, z])

            mask = binary_closing(mask, structure=struct_3d_26, iterations=2)
            labeled_mask, num_features = label(mask, structure=struct_3d_26)
            if num_features > 0:
                sizes = np.bincount(labeled_mask.ravel())
                sizes[0] = 0
                mask = labeled_mask == np.argmax(sizes)

        # Class 3: Trachea (Continuous central airway)
        elif c == 3:
            for z in range(mask.shape[2]):
                if mask[:, :, z].any():
                    mask[:, :, z] = binary_fill_holes(mask[:, :, z])

            labeled_mask, num_features = label(mask, structure=struct_3d_26)
            if num_features > 0:
                sizes = np.bincount(labeled_mask.ravel())
                sizes[0] = 0
                mask = labeled_mask == np.argmax(sizes)

            z_kernel = np.zeros((1, 1, 3), dtype=bool)
            z_kernel[0, 0, :] = True
            mask = binary_closing(mask, structure=z_kernel)

        # Class 4: Aorta (Ascending/Descending sections)
        elif c == 4:
            for z in range(mask.shape[2]):
                if mask[:, :, z].any():
                    slice_2d = binary_closing(
                        mask[:, :, z], structure=struct_2d_8, iterations=1
                    )
                    mask[:, :, z] = binary_fill_holes(slice_2d)

            labeled_mask, num_features = label(mask, structure=struct_3d_26)
            if num_features > 0:
                sizes = np.bincount(labeled_mask.ravel())
                sizes[0] = 0
                valid_labels = np.where(sizes >= 1500)[0]
                if len(valid_labels) > 0:
                    mask = np.isin(labeled_mask, valid_labels)
                else:
                    mask = labeled_mask == np.argmax(sizes)

        empty_space = cleaned_arr[:, :, z_min:z_max] == 0
        cleaned_arr[:, :, z_min:z_max][mask & empty_space] = c

    return cleaned_arr


# ==============================================================================
# 2. 2D SLICE STITCHER TO SITK IMAGE
# ==============================================================================
def stitch_2d_slices_to_sitk(
    patient_id: str,
    slice_files: List[Path],
    ref_nifti_path: Path,
    num_classes: int = 5,
) -> sitk.Image:
    """Stitches 2D slice predictions into a 3D SimpleITK image using the reference volume's spatial metadata."""
    ref_sitk = sitk.ReadImage(str(ref_nifti_path))
    # SimpleITK GetSize() returns (X, Y, Z) / (Width, Height, Depth)
    target_X, target_Y, target_Z = ref_sitk.GetSize()

    # Sort slice files numerically by Z index in filename
    sorted_files = sorted(
        slice_files, key=lambda p: int(re.search(r"(\d+)(?=\.[^.]+$)", p.name).group(1))
    )

    if len(sorted_files) != target_Z:
        raise ValueError(
            f"Slice count mismatch for {patient_id}: Found {len(sorted_files)} slice files, "
            f"but reference volume Z-depth is {target_Z}."
        )

    # SimpleITK's GetImageFromArray expects numpy shape: (Z, Y, X) -> (Depth, Height, Width)
    vol_arr = np.zeros((target_Z, target_Y, target_X), dtype=np.uint8)

    for z_idx, slice_path in enumerate(sorted_files):
        if slice_path.suffix == ".npy":
            slice_data = np.load(slice_path)
        else:
            slice_data = imread(slice_path)
            if slice_data.max() > num_classes:
                slice_data = (slice_data // 63).astype(np.uint8)

        # Ensure slice matches 2D dimensions (Y, X) / (H, W)
        if slice_data.shape != (target_Y, target_X):
            slice_data = resize(
                slice_data,
                (target_Y, target_X),
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(np.uint8)

        # Place slice directly at Z index
        vol_arr[z_idx, :, :] = slice_data

    # Convert NumPy array (Z, Y, X) directly to SimpleITK (X, Y, Z)
    out_sitk = sitk.GetImageFromArray(vol_arr)
    out_sitk.CopyInformation(ref_sitk)
    return out_sitk


# ==============================================================================
# 3. PHYSICAL RESAMPLING TO RAW REFERENCE CT
# ==============================================================================
def resample_to_raw_reference(pred_sitk: sitk.Image, raw_ct_path: Path) -> sitk.Image:
    """Resamples a 3D predicted SimpleITK image to match raw CT physical reference."""
    raw_sitk = sitk.ReadImage(str(raw_ct_path))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(raw_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)

    resampled = resampler.Execute(pred_sitk)
    return sitk.Cast(resampled, sitk.sitkUInt8)


# ==============================================================================
# 4. PATIENT WORKFLOW EXECUTION
# ==============================================================================
def process_patient(
    patient_id: str,
    pred_dir: Path,
    raw_dir: Path,
    preprocessed_dir: Optional[Path],
    output_dir: Path,
    num_classes: int,
    apply_post: bool,
    is_2d_input: bool,
) -> None:
    # 1. Locate Raw CT Reference File
    raw_candidates = [
        raw_dir / "train" / patient_id / f"{patient_id}.nii.gz",
        raw_dir / "test" / patient_id / f"{patient_id}.nii.gz",
        raw_dir / "test" / f"{patient_id}.nii.gz",
        raw_dir / patient_id / f"{patient_id}.nii.gz",
    ]
    raw_ct_path = next((p for p in raw_candidates if p.exists()), None)
    if raw_ct_path is None:
        print(f"[Warning] Raw CT reference not found for {patient_id}")
        return

    # 2. Load or Stitch Prediction to 3D SimpleITK Image
    if is_2d_input:
        if preprocessed_dir is None:
            raise ValueError(
                "`--preprocessed_dir` must be provided when processing 2D slice predictions."
            )

        prep_candidates = [
            preprocessed_dir / "train" / patient_id / f"{patient_id}.nii.gz",
            preprocessed_dir / "val" / patient_id / f"{patient_id}.nii.gz",
            preprocessed_dir / "test" / patient_id / f"{patient_id}.nii.gz",
            preprocessed_dir / patient_id / f"{patient_id}.nii.gz",
        ]
        prep_ct_path = next((p for p in prep_candidates if p.exists()), None)
        if prep_ct_path is None:
            print(f"[Warning] Preprocessed NIfTI reference not found for {patient_id}")
            return

        slice_files = list(pred_dir.glob(f"{patient_id}_*"))
        pred_sitk = stitch_2d_slices_to_sitk(
            patient_id, slice_files, prep_ct_path, num_classes=num_classes
        )
    else:
        # 3D Model Output Input
        pred_candidates = [
            pred_dir / patient_id / f"{patient_id}.nii.gz",
            pred_dir / patient_id / "pred.nii.gz",
            pred_dir / f"{patient_id}.nii.gz",
        ]
        pred_path = next((p for p in pred_candidates if p.exists()), None)
        if pred_path is None:
            print(f"[Warning] 3D prediction file not found for {patient_id}")
            return
        pred_sitk = sitk.ReadImage(str(pred_path))

    # 3. Apply 3D Morphological Post-Processing
    if apply_post:
        # sitk.GetArrayFromImage returns numpy shape (Z, Y, X)
        arr_zyx = sitk.GetArrayFromImage(pred_sitk).astype(np.uint8)

        # Transpose to (Y, X, Z) or (H, W, Z) for post_process_3d
        arr_hwz = np.transpose(arr_zyx, (1, 2, 0))

        cleaned_hwz = post_process_3d(arr_hwz, num_classes=num_classes)

        # Transpose back to (Z, Y, X) for SimpleITK
        cleaned_zyx = np.transpose(cleaned_hwz, (2, 0, 1))

        cleaned_sitk = sitk.GetImageFromArray(cleaned_zyx)
        cleaned_sitk.CopyInformation(pred_sitk)
        pred_sitk = cleaned_sitk

    # 4. Resample back to original Raw CT Reference Space
    final_sitk = resample_to_raw_reference(pred_sitk, raw_ct_path)

    # 5. Write final output NIfTI
    out_patient_dir = output_dir / patient_id
    out_patient_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(final_sitk, str(out_patient_dir / f"{patient_id}.nii.gz"))


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Universal post-processor and raw reference resampler for 2D & 3D predictions."
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        required=True,
        help="Directory containing predictions (2D slice files or 3D NIfTI subfolders)",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help="Directory containing raw original CT volumes",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output destination directory for resampled predictions",
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=Path,
        default=None,
        help="Directory containing 3D preprocessed NIfTI files (Required if using 2D inputs)",
    )
    parser.add_argument(
        "--is_2d_input",
        action="store_true",
        help="Set if predictions are 2D slice files (.npy / .png) that require stitching",
    )
    parser.add_argument(
        "--num_classes", type=int, default=5, help="Number of segmentation classes"
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Enable organ-specific 3D morphological post-processing",
    )
    parser.add_argument(
        "-p",
        "--process",
        type=int,
        default=-1,
        help="Multiprocessing workers (-1 for all cores)",
    )

    args = parser.parse_args()

    # Discover unique Patient IDs
    patient_ids = set()
    if args.is_2d_input:
        regex = re.compile(r"^(Patient_\d+)")
        for f in args.pred_dir.glob("*"):
            match = regex.match(f.name)
            if match:
                patient_ids.add(match.group(1))
    else:
        for p in args.pred_dir.glob("*"):
            if p.is_dir():
                patient_ids.add(p.name)
            elif p.name.endswith(".nii.gz"):
                patient_ids.add(p.name.replace(".nii.gz", ""))

    patient_list = sorted(list(patient_ids))
    print(f">> Processing {len(patient_list)} patients from {args.pred_dir}...")

    pfun = partial(
        process_patient,
        pred_dir=args.pred_dir,
        raw_dir=args.raw_dir,
        preprocessed_dir=args.preprocessed_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        apply_post=args.post,
        is_2d_input=args.is_2d_input,
    )

    if args.process == 1:
        for pid in tqdm(patient_list):
            pfun(pid)
    else:
        num_workers = None if args.process == -1 else args.process
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap(pfun, patient_list), total=len(patient_list)))

    print(f">> Post-processing and raw resampling complete! Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
