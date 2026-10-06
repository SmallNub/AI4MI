#!/usr/bin/env python3
"""
Universal Post-Processing & Spatial Remapping Pipeline
------------------------------------------------------
1. Stitches 2D slice predictions (.png / .npy) or reads 3D model predictions in preprocessed space.
2. Applies organ-specific 3D post-processing in native (X, Y, Z) array space.
3. Maps predictions back to the preprocessed CT NIfTI header/affine.
4. Resamples the segmentation back to the original raw CT physical space using SimpleITK.
"""

import argparse
import re
import shutil
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

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
# 1. ORGAN-SPECIFIC 3D POST-PROCESSING (Native X, Y, Z Layout)
# ==============================================================================
def post_process_3d(
    arr: np.ndarray,
    num_classes: int = 5,
    crop_z_margins: bool = False,
) -> np.ndarray:
    """
    Organ-specific 3D post-processing tailored for axial CT volumes.
    Iterates sequentially through class indices 1 to 4.
    Input shape expected: (X, Y, Z) where Z is the axial slice index.

    Classes:
        1: Esophagus
        2: Heart
        3: Trachea
        4: Aorta
    """
    cleaned_arr = np.zeros_like(arr, dtype=np.uint8)
    struct_3d_26 = generate_binary_structure(3, 3)
    struct_2d_8 = generate_binary_structure(2, 2)

    X, Y, Z = arr.shape
    z_min, z_max = 0, Z
    if crop_z_margins:
        z_min = int(Z * 0.05)
        z_max = int(Z * 0.95)

    for c in range(1, num_classes):
        mask = arr[:, :, z_min:z_max] == c
        if not mask.any():
            continue

        # Class 1: Esophagus (Thin vertical structure across axial slices)
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

        # Class 4: Aorta (Ascending/Descending tubular sections)
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
# 2. HELPER FUNCTIONS
# ==============================================================================
def get_z_index(filepath: Path) -> int:
    """Extracts axial slice integer z-index from file name."""
    match = re.search(r"_(\d+)\.[^.]+$", filepath.name)
    if match:
        return int(match.group(1))
    return int(filepath.stem.split("_")[-1])


def resample_mask_to_raw_ct(
    segmented_nii_path: Path, raw_ct_path: Path, output_path: Path
):
    """
    Resamples a 3D segmentation volume in preprocessed physical space back
    to match the original raw CT physical spatial resolution, origin, and bounds.
    """
    seg_sitk = sitk.ReadImage(str(segmented_nii_path))
    raw_sitk = sitk.ReadImage(str(raw_ct_path))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(raw_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)

    resampled = resampler.Execute(seg_sitk)
    sitk.WriteImage(sitk.Cast(resampled, sitk.sitkUInt8), str(output_path))


# ==============================================================================
# 3. 2D & 3D PATIENT MERGING PIPELINE
# ==============================================================================
def process_patient(
    patient_id: str,
    pred_dir: Path,
    preprocessed_scan_pattern: str,
    raw_scan_pattern: str,
    output_dir: Path,
    num_classes: int = 5,
    apply_post: bool = True,
    is_2d_input: bool = True,
) -> None:
    # 1. Load Preprocessed Reference Volume (X, Y, Z) via Nibabel
    prep_nii_path = Path(preprocessed_scan_pattern.format(id_=patient_id))
    if not prep_nii_path.exists():
        print(f"[Warning] Preprocessed reference NIfTI not found: {prep_nii_path}")
        return

    orig_nib = nib.load(str(prep_nii_path))
    X, Y, Z = orig_nib.shape

    out_nii_path = output_dir / f"{patient_id}.nii.gz"

    if is_2d_input:
        # Collect and sort 2D slice files for this patient
        slice_files = [
            p
            for p in pred_dir.glob(f"{patient_id}_*")
            if p.suffix in [".png", ".npy", ".jpg", ".tif"]
        ]
        if not slice_files:
            print(f"[Warning] No 2D slice files found for patient {patient_id}")
            return

        assert (
            len(slice_files) == Z
        ), f"Slice count mismatch for patient {patient_id}: prep scan Z={Z}, slices={len(slice_files)}"

        res_arr = np.zeros((X, Y, Z), dtype=np.uint8)

        for slice_path in slice_files:
            z = get_z_index(slice_path)
            if slice_path.suffix == ".npy":
                img_arr = np.load(slice_path)
            else:
                img_arr = imread(slice_path)

            if img_arr.max() > num_classes:
                img_arr = (img_arr // 63).astype(np.uint8)

            # Resize to preprocessed target (X, Y) using nearest-neighbor interpolation
            if img_arr.shape != (X, Y):
                resized = resize(
                    img_arr,
                    (X, Y),
                    mode="constant",
                    preserve_range=True,
                    anti_aliasing=False,
                    order=0,
                ).astype(np.uint8)
            else:
                resized = img_arr.astype(np.uint8)

            res_arr[:, :, z] = resized

    else:
        # 3D Model NIfTI input
        pred_3d_path = pred_dir / f"{patient_id}.nii.gz"
        if not pred_3d_path.exists():
            pred_3d_path = pred_dir / patient_id / f"{patient_id}.nii.gz"

        if not pred_3d_path.exists():
            print(f"[Warning] 3D prediction NIfTI not found for {patient_id}")
            return

        pred_nib = nib.load(str(pred_3d_path))
        res_arr = np.asarray(pred_nib.dataobj, dtype=np.uint8)

    # 2. Apply Organ-Specific Post-Processing in (X, Y, Z) Space
    if apply_post:
        res_arr = post_process_3d(res_arr, num_classes=num_classes)

    # 3. Save initial volume using preprocessed affine matrix & header
    new_nib = nib.nifti1.Nifti1Image(
        res_arr, affine=orig_nib.affine, header=orig_nib.header
    )
    nib.save(new_nib, str(out_nii_path))

    # 4. Physical Resampling to Raw Scan Space
    raw_ct_path = Path(raw_scan_pattern.format(id_=patient_id))
    if not raw_ct_path.exists():
        print(
            f"[Warning] Raw CT scan reference not found: {raw_ct_path}. Skipping resampling."
        )
        return

    # Resample from preprocessed physical space back to raw physical space
    resample_mask_to_raw_ct(out_nii_path, raw_ct_path, out_nii_path)


# ==============================================================================
# MAIN CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Universal post-processing and raw physical resampling pipeline."
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        required=True,
        help="Directory containing slice predictions (.png/.npy) or 3D NIfTIs",
    )
    parser.add_argument(
        "--preprocessed_scan_pattern",
        type=str,
        required=True,
        help="Path pattern to preprocessed scan (e.g. 'data/preprocessed/{id_}.nii.gz')",
    )
    parser.add_argument(
        "--raw_scan_pattern",
        type=str,
        required=True,
        help="Path pattern to original raw CT scan (e.g. 'data/raw/{id_}.nii.gz')",
    )
    parser.add_argument(
        "--dest_folder",
        type=Path,
        required=True,
        help="Output directory for saved .nii.gz volumes",
    )
    parser.add_argument(
        "--grp_regex",
        type=str,
        default=r"^(Patient_\d+)",
        help="Regex pattern to extract patient IDs from slice filenames",
    )
    parser.add_argument(
        "--is_2d_input",
        action="store_true",
        help="Flag indicating predictions are 2D slice files",
    )
    parser.add_argument(
        "--num_classes", type=int, default=5, help="Number of segmentation classes"
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Enable 3D organ-specific post-processing",
    )
    parser.add_argument(
        "-p",
        "--process",
        type=int,
        default=-1,
        help="Multiprocessing workers (-1 for all cores)",
    )

    args = parser.parse_args()

    args.dest_folder.mkdir(parents=True, exist_ok=True)

    patient_ids = set()
    if args.is_2d_input:
        regex = re.compile(args.grp_regex)
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
        preprocessed_scan_pattern=args.preprocessed_scan_pattern,
        raw_scan_pattern=args.raw_scan_pattern,
        output_dir=args.dest_folder,
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

    print(f">> Pipeline complete! Restored volumes saved to {args.dest_folder}")


if __name__ == "__main__":
    main()
