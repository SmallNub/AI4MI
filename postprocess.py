#!/usr/bin/env python3
"""
Universal Post-Processing & Spatial Remapping Pipeline
------------------------------------------------------
1. Stitches 2D slice predictions (.png / .npy) or reads 3D model predictions in preprocessed space.
2. If post-processing is enabled:
   - Evaluates all candidate removal-only 3D post-processing policies against Ground Truth (if provided) using Mean Dice Score.
   - Selects the best-performing policy and applies it.
3. Maps predictions back to the preprocessed CT NIfTI header/affine.
4. Resamples the segmentation back to the original raw CT physical space using SimpleITK.
"""

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

# ==============================================================================
# 1. ORGAN-SPECIFIC 3D POST-PROCESSING & POLICIES
# ==============================================================================

LABELS = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}
POLICY_FIELDS = {
    "enabled",
    "connectivity",
    "keep_largest",
    "min_component_mm3",
    "min_relative_to_largest",
    "slice_lcc_axis",
    "slice_connectivity",
}

# Base policy: no changes.
BASE_POLICY: Dict[str, Dict[str, Any]] = {
    str(label): {
        "enabled": False,
        "connectivity": 3,  # 1/2/3 means 6/18/26-neighbour 3D connectivity.
        "keep_largest": False,
        "min_component_mm3": 0.0,
        "min_relative_to_largest": 0.0,
        "slice_lcc_axis": None,
        "slice_connectivity": 2,  # 1/2 means 4/8-neighbour connectivity in 2D.
    }
    for label in LABELS
}


def _with_overrides(
    policy: Dict[str, Dict[str, Any]], **overrides: Any
) -> Dict[str, Dict[str, Any]]:
    """Return a copied policy with field overrides for selected labels."""
    copied = deepcopy(policy)
    for label, values in overrides.items():
        copied[label].update(values)
    return copied


BUILTIN_POLICIES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "none": deepcopy(BASE_POLICY),
    "lcc_3d_non_esophagus": _with_overrides(
        BASE_POLICY,
        **{
            "2": {"enabled": True, "keep_largest": True},
            "3": {"enabled": True, "keep_largest": True},
            "4": {"enabled": True, "keep_largest": True},
        },
    ),
    "esophagus_min_500mm3": _with_overrides(
        BASE_POLICY,
        **{
            "1": {
                "enabled": True,
                "min_component_mm3": 500.0,
            }
        },
    ),
    "heart_3d_lcc_axial_2d_lcc": _with_overrides(
        BASE_POLICY,
        **{
            "2": {
                "enabled": True,
                "keep_largest": True,
                "slice_lcc_axis": 2,
                "slice_connectivity": 2,
            }
        },
    ),
    "relative_20pct_all": _with_overrides(
        BASE_POLICY,
        **{
            str(label): {
                "enabled": True,
                "min_relative_to_largest": 0.2,
            }
            for label in LABELS
        },
    ),
}


def component_statistics(
    mask: np.ndarray, connectivity: int
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Label 3D components and return labels, count, and voxel counts."""
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D binary mask, got shape {mask.shape}")
    if connectivity not in (1, 2, 3):
        raise ValueError("3D connectivity must be 1, 2, or 3")

    structure = ndimage.generate_binary_structure(3, connectivity)
    labelled, n_components = ndimage.label(mask, structure=structure)
    counts = np.bincount(labelled.ravel(), minlength=n_components + 1)
    counts[0] = 0
    return labelled, n_components, counts


def retain_largest_component_per_slice(
    mask: np.ndarray, axis: int, connectivity: int
) -> np.ndarray:
    """Retain largest 2D component independently in each slice."""
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D binary mask, got shape {mask.shape}")
    if axis not in (0, 1, 2):
        raise ValueError("slice_lcc_axis must be 0, 1, 2, or null")
    if connectivity not in (1, 2):
        raise ValueError("slice_connectivity must be 1 or 2")

    moved = np.moveaxis(mask, axis, 0)
    result = np.zeros_like(moved, dtype=bool)
    structure = ndimage.generate_binary_structure(2, connectivity)

    for index, slice_mask in enumerate(moved):
        labelled, n_components = ndimage.label(slice_mask, structure=structure)
        if n_components == 0:
            continue
        counts = np.bincount(labelled.ravel(), minlength=n_components + 1)
        counts[0] = 0
        largest_id = int(np.argmax(counts))
        result[index] = labelled == largest_id

    return np.moveaxis(result, 0, axis)


def validate_policy(policy_by_label: Dict[str, Dict[str, Any]]) -> None:
    """Reject malformed policy values before any prediction volume is modified."""
    if set(policy_by_label) != set(BASE_POLICY):
        raise ValueError("policy must define exactly labels '1', '2', '3' and '4'")

    for label, policy in policy_by_label.items():
        unknown = set(policy) - POLICY_FIELDS
        if unknown:
            raise ValueError(f"unknown policy fields for label {label}: {sorted(unknown)}")
        if int(policy["connectivity"]) not in (1, 2, 3):
            raise ValueError(f"Label {label}: connectivity must be 1, 2 or 3")
        if int(policy["slice_connectivity"]) not in (1, 2):
            raise ValueError(f"Label {label}: slice_connectivity must be 1 or 2")
        axis = policy["slice_lcc_axis"]
        if axis is not None and int(axis) not in (0, 1, 2):
            raise ValueError(f"Label {label}: slice_lcc_axis must be 0, 1, 2 or null")
        if float(policy["min_component_mm3"]) < 0:
            raise ValueError(f"Label {label}: min_component_mm3 must be non-negative")
        relative = float(policy["min_relative_to_largest"])
        if not 0.0 <= relative <= 1.0:
            raise ValueError(
                f"Label {label}: min_relative_to_largest must be between 0 and 1"
            )


def apply_policy(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    policy: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply one policy to a class mask."""
    connectivity = int(policy["connectivity"])
    labelled, n_before, counts_before = component_statistics(mask, connectivity)
    voxel_volume_mm3 = float(np.prod(spacing))
    input_voxels = int(mask.sum())

    report: Dict[str, Any] = {
        "components_before": n_before,
        "largest_component_mm3_before": (
            float(counts_before.max() * voxel_volume_mm3) if n_before else 0.0
        ),
        "input_voxels": input_voxels,
        "removed_by_3d_filter_voxels": 0,
        "removed_by_slice_lcc_voxels": 0,
        "slice_lcc_axis": policy["slice_lcc_axis"],
    }

    if not bool(policy["enabled"]):
        report.update(
            {
                "components_after": n_before,
                "output_voxels": input_voxels,
                "removed_voxels": 0,
                "added_voxels": 0,
            }
        )
        return mask.copy(), report

    keep_ids = np.arange(1, n_before + 1, dtype=np.int32)

    if n_before and bool(policy["keep_largest"]):
        keep_ids = np.array([int(np.argmax(counts_before))], dtype=np.int32)

    min_component_mm3 = float(policy["min_component_mm3"])
    if n_before and min_component_mm3 > 0:
        component_volumes_mm3 = counts_before * voxel_volume_mm3
        absolute_ids = np.flatnonzero(component_volumes_mm3 >= min_component_mm3)
        absolute_ids = absolute_ids[absolute_ids != 0]
        keep_ids = np.intersect1d(keep_ids, absolute_ids)

    min_relative = float(policy["min_relative_to_largest"])
    if n_before and min_relative > 0:
        relative_ids = np.flatnonzero(
            counts_before >= min_relative * counts_before.max()
        )
        relative_ids = relative_ids[relative_ids != 0]
        keep_ids = np.intersect1d(keep_ids, relative_ids)

    cleaned = np.isin(labelled, keep_ids)
    report["removed_by_3d_filter_voxels"] = int(
        np.logical_and(mask, ~cleaned).sum()
    )

    slice_axis = policy["slice_lcc_axis"]
    if slice_axis is not None:
        before_slice_lcc = cleaned.copy()
        cleaned = retain_largest_component_per_slice(
            cleaned,
            axis=int(slice_axis),
            connectivity=int(policy["slice_connectivity"]),
        )
        report["removed_by_slice_lcc_voxels"] = int(
            np.logical_and(before_slice_lcc, ~cleaned).sum()
        )

    if np.any(cleaned & ~mask):
        raise RuntimeError("Internal error: a removal-only policy added voxels")

    _, n_after, _ = component_statistics(cleaned, connectivity)
    output_voxels = int(cleaned.sum())
    report.update(
        {
            "components_after": n_after,
            "output_voxels": output_voxels,
            "removed_voxels": input_voxels - output_voxels,
            "added_voxels": 0,
        }
    )
    return cleaned.astype(bool), report


def postprocess_volume(
    labels: np.ndarray,
    spacing: Tuple[float, float, float],
    policy_by_label: Dict[str, Dict[str, Any]],
    label_order: List[int],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Apply per-class policies to a 3D label map."""
    if labels.ndim != 3:
        raise ValueError(f"Expected one 3D label map, got shape {labels.shape}")

    unique = set(np.unique(labels).astype(int))
    allowed = {0, *LABELS}
    if not unique.issubset(allowed):
        raise ValueError(f"Expected labels {sorted(allowed)}, found {sorted(unique)}")
    if sorted(label_order) != sorted(LABELS):
        raise ValueError("label_order must contain labels 1, 2, 3, and 4 exactly once")

    validate_policy(policy_by_label)
    source = labels.astype(np.uint8, copy=True)
    output = source.copy()
    records: List[Dict[str, Any]] = []

    for class_id in label_order:
        original_mask = source == class_id
        cleaned_mask, report = apply_policy(
            original_mask,
            spacing,
            policy_by_label[str(class_id)],
        )

        output[original_mask & ~cleaned_mask] = 0

        records.append(
            {
                "class": class_id,
                "organ": LABELS[class_id],
                "enabled": bool(policy_by_label[str(class_id)]["enabled"]),
                "keep_largest": bool(policy_by_label[str(class_id)]["keep_largest"]),
                "min_component_mm3": float(
                    policy_by_label[str(class_id)]["min_component_mm3"]
                ),
                "min_relative_to_largest": float(
                    policy_by_label[str(class_id)]["min_relative_to_largest"]
                ),
                **report,
            }
        )

    return output, records


def load_policy(
    preset: str, config_path: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """Load policy preset and apply optional JSON config overrides."""
    if preset not in BUILTIN_POLICIES:
        raise ValueError(
            f"Unknown preset {preset!r}; choose from {sorted(BUILTIN_POLICIES)}"
        )

    policy = deepcopy(BUILTIN_POLICIES[preset])
    if config_path is None:
        validate_policy(policy)
        return policy

    with config_path.open() as file:
        supplied = json.load(file)
    if not isinstance(supplied, dict):
        raise ValueError("Configuration must be a JSON object keyed by label strings")

    for label, overrides in supplied.items():
        if label not in policy:
            raise ValueError(f"Unknown label {label!r}; use strings '1' through '4'")
        if not isinstance(overrides, dict):
            raise ValueError(f"Policy for label {label} must be a JSON object")
        unknown = set(overrides) - POLICY_FIELDS
        if unknown:
            raise ValueError(f"Unknown fields for label {label}: {sorted(unknown)}")
        policy[label].update(overrides)

    validate_policy(policy)
    return policy


# ==============================================================================
# 2. EVALUATION METRICS
# ==============================================================================
def compute_dice_score(
    pred: np.ndarray, gt: np.ndarray, labels: List[int] = [1, 2, 3, 4]
) -> Dict[int, float]:
    """Compute per-class Dice coefficient between predicted and ground truth masks."""
    dice_scores = {}
    for c in labels:
        p_mask = pred == c
        g_mask = gt == c
        intersection = np.logical_and(p_mask, g_mask).sum()
        total_voxels = p_mask.sum() + g_mask.sum()
        if total_voxels == 0:
            dice_scores[c] = 1.0
        else:
            dice_scores[c] = float((2.0 * intersection) / total_voxels)
    return dice_scores


def evaluate_policies(
    labels: np.ndarray,
    spacing: Tuple[float, float, float],
    gt_labels: Optional[np.ndarray] = None,
    candidate_presets: Optional[List[str]] = None,
    label_order: Optional[List[int]] = None,
) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    """Loop through candidate post-processing policies and evaluate each method.

    If ground truth is provided, evaluates methods using Mean Dice Score and
    selects the best-performing policy.
    If ground truth is not provided, defaults to comparing connectivity and
    noise reduction metrics to pick the safest filtering configuration.
    """
    if candidate_presets is None:
        candidate_presets = list(BUILTIN_POLICIES.keys())
    if label_order is None:
        label_order = [1, 2, 3, 4]

    best_preset = candidate_presets[0]
    best_score = -1.0
    best_processed_arr = labels.copy()
    best_records: Dict[str, Any] = {}

    eval_results = []

    for preset_name in candidate_presets:
        policy = load_policy(preset_name)
        processed_arr, records = postprocess_volume(
            labels=labels,
            spacing=spacing,
            policy_by_label=policy,
            label_order=label_order,
        )

        if gt_labels is not None:
            # Evaluation with Ground Truth via Mean Dice Score
            dice_dict = compute_dice_score(processed_arr, gt_labels, labels=label_order)
            mean_dice = float(np.mean(list(dice_dict.values())))
            score = mean_dice
            eval_results.append(
                {
                    "preset": preset_name,
                    "mean_dice": mean_dice,
                    "per_class_dice": dice_dict,
                }
            )
        else:
            # Evaluation without Ground Truth (e.g., component count & noise filtering ratio)
            total_removed = sum(r.get("removed_voxels", 0) for r in records)
            total_input = sum(r.get("input_voxels", 1) for r in records) or 1
            removal_ratio = total_removed / total_input
            # Score penalizes over-removal (>15% voxels) while favoring artifact removal
            score = 1.0 - abs(removal_ratio - 0.02)
            eval_results.append(
                {
                    "preset": preset_name,
                    "removal_ratio": removal_ratio,
                    "score": score,
                }
            )

        if score > best_score:
            best_score = score
            best_preset = preset_name
            best_processed_arr = processed_arr
            best_records = {
                "preset": preset_name,
                "score": score,
                "records": records,
                "all_evaluations": eval_results,
            }

    return best_processed_arr, best_preset, best_score, best_records


def load_patient_prediction(
    patient_id: str,
    pred_dir: Path,
    preprocessed_scan_pattern: str,
    is_2d_input: bool,
    num_classes: int,
) -> Tuple[nib.nifti1.Nifti1Image, np.ndarray]:
    """Load one patient prediction in the preprocessed reference geometry."""
    prep_nii_path = Path(preprocessed_scan_pattern.format(id_=patient_id))
    if not prep_nii_path.exists():
        raise FileNotFoundError(f"Preprocessed reference NIfTI not found: {prep_nii_path}")

    reference = nib.load(str(prep_nii_path))
    X, Y, Z = reference.shape

    if is_2d_input:
        slice_files = [
            path
            for path in pred_dir.glob(f"{patient_id}_*")
            if path.suffix in [".png", ".npy", ".jpg", ".tif"]
        ]
        if not slice_files:
            raise FileNotFoundError(f"No 2D prediction slices found for {patient_id}")
        if len(slice_files) != Z:
            raise ValueError(
                f"Slice count mismatch for {patient_id}: reference Z={Z}, "
                f"prediction slices={len(slice_files)}"
            )

        prediction = np.zeros((X, Y, Z), dtype=np.uint8)
        for slice_path in slice_files:
            z = get_z_index(slice_path)
            if not 0 <= z < Z:
                raise ValueError(f"Slice index {z} out of bounds for {patient_id} (Z={Z})")
            img_arr = np.load(slice_path) if slice_path.suffix == ".npy" else imread(slice_path)
            if img_arr.max() > num_classes:
                img_arr = (img_arr // 63).astype(np.uint8)
            if img_arr.shape != (X, Y):
                img_arr = resize(
                    img_arr,
                    (X, Y),
                    mode="constant",
                    preserve_range=True,
                    anti_aliasing=False,
                    order=0,
                )
            prediction[:, :, z] = img_arr.astype(np.uint8)
    else:
        prediction_path = pred_dir / f"{patient_id}.nii.gz"
        if not prediction_path.exists():
            prediction_path = pred_dir / patient_id / f"{patient_id}.nii.gz"
        if not prediction_path.exists():
            raise FileNotFoundError(f"3D prediction NIfTI not found for {patient_id}")
        prediction = np.asarray(nib.load(str(prediction_path)).dataobj, dtype=np.uint8)

    if prediction.shape != reference.shape:
        raise ValueError(
            f"Prediction/reference shape mismatch for {patient_id}: "
            f"{prediction.shape} vs {reference.shape}"
        )
    return reference, prediction


def select_global_policy(
    patient_ids: List[str],
    pred_dir: Path,
    preprocessed_scan_pattern: str,
    gt_scan_pattern: str,
    is_2d_input: bool,
    num_classes: int,
    label_order: List[int],
) -> Dict[str, Any]:
    """Select one policy by mean patient/class Dice over the validation set."""
    if not patient_ids:
        raise ValueError("No patients available for global policy evaluation")

    preset_scores: Dict[str, List[float]] = {
        preset: [] for preset in BUILTIN_POLICIES
    }
    class_scores: Dict[str, Dict[int, List[float]]] = {
        preset: {class_id: [] for class_id in label_order}
        for preset in BUILTIN_POLICIES
    }

    for patient_id in tqdm(patient_ids, desc="Evaluating policies", unit="patient"):
        reference, prediction = load_patient_prediction(
            patient_id,
            pred_dir,
            preprocessed_scan_pattern,
            is_2d_input,
            num_classes,
        )
        gt_path = Path(gt_scan_pattern.format(id_=patient_id))
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found for {patient_id}: {gt_path}")
        target = np.asarray(nib.load(str(gt_path)).dataobj, dtype=np.uint8)
        if target.shape != prediction.shape:
            raise ValueError(
                f"Prediction/ground-truth shape mismatch for {patient_id}: "
                f"{prediction.shape} vs {target.shape}"
            )

        spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
        for preset_name in BUILTIN_POLICIES:
            processed, _ = postprocess_volume(
                labels=prediction,
                spacing=spacing,
                policy_by_label=load_policy(preset_name),
                label_order=label_order,
            )
            dice_by_class = compute_dice_score(processed, target, labels=label_order)
            for class_id, score in dice_by_class.items():
                class_scores[preset_name][class_id].append(score)
                preset_scores[preset_name].append(score)

    comparisons = []
    for preset_name in BUILTIN_POLICIES:
        per_class = {
            str(class_id): float(np.mean(scores))
            for class_id, scores in class_scores[preset_name].items()
        }
        comparisons.append(
            {
                "preset": preset_name,
                "mean_patient_class_dice": float(np.mean(preset_scores[preset_name])),
                "per_class_mean_dice": per_class,
            }
        )

    winner = max(comparisons, key=lambda item: item["mean_patient_class_dice"])
    selected_preset = winner["preset"]
    return {
        "selection_method": "mean_patient_class_dice",
        "selected_preset": selected_preset,
        "selected_policy": load_policy(selected_preset),
        "selected_mean_patient_class_dice": winner["mean_patient_class_dice"],
        "evaluated_patients": patient_ids,
        "candidate_results": comparisons,
    }


# ==============================================================================
# 3. HELPER FUNCTIONS
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
    """Resamples a 3D segmentation volume in preprocessed physical space back

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
# 4. 2D & 3D PATIENT MERGING PIPELINE
# ==============================================================================
def process_patient(
    patient_id: str,
    pred_dir: Path,
    preprocessed_scan_pattern: str,
    raw_scan_pattern: str,
    output_dir: Path,
    gt_scan_pattern: Optional[str] = None,
    num_classes: int = 5,
    apply_post: bool = True,
    is_2d_input: bool = True,
    preset: str = "lcc_3d_non_esophagus",
    config_path: Optional[Path] = None,
    label_order: Optional[List[int]] = None,
    selected_policy: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    # 1. Load Preprocessed Reference Volume (X, Y, Z) via Nibabel
    try:
        orig_nib, res_arr = load_patient_prediction(
            patient_id,
            pred_dir,
            preprocessed_scan_pattern,
            is_2d_input,
            num_classes,
        )
    except FileNotFoundError as error:
        print(f"[Warning] {error}")
        return

    out_nii_path = output_dir / f"{patient_id}.nii.gz"

    # 2. Evaluate & Apply Post-Processing Policies
    if apply_post:
        spacing = tuple(float(v) for v in orig_nib.header.get_zooms()[:3])
        policy = selected_policy or load_policy(preset, config_path)
        res_arr, _ = postprocess_volume(
            labels=res_arr,
            spacing=spacing,
            policy_by_label=policy,
            label_order=label_order or [1, 2, 3, 4],
        )

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
        "--gt_scan_pattern",
        type=str,
        default=None,
        help="Optional path pattern to Ground Truth NIfTI scans for evaluating post-processing policies (e.g. 'data/gt/{id_}.nii.gz')",
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
        "--evaluate_all_policies",
        action="store_true",
        help=(
            "Evaluate every built-in policy across all supplied patients using "
            "mean patient/class Dice, save the global winner, and apply it"
        ),
    )
    parser.add_argument(
        "--policy-selection",
        type=Path,
        default=None,
        help=(
            "Load a policy selection JSON previously created by "
            "--evaluate_all_policies (use this for test inference)"
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(BUILTIN_POLICIES),
        default="lcc_3d_non_esophagus",
        help="Named policy preset for 3D post-processing when not evaluating all policies.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file for custom policy overrides.",
    )
    parser.add_argument(
        "--label_order",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Class processing order; must contain labels 1, 2, 3 and 4 once each.",
    )
    parser.add_argument(
        "-p",
        "--process",
        type=int,
        default=-1,
        help="Multiprocessing workers (-1 for all cores)",
    )

    args = parser.parse_args()

    if args.evaluate_all_policies and not args.post:
        parser.error("--evaluate_all_policies requires --post")
    if args.evaluate_all_policies and not args.gt_scan_pattern:
        parser.error("--evaluate_all_policies requires --gt_scan_pattern")
    if args.evaluate_all_policies and args.policy_selection:
        parser.error("Use either --evaluate_all_policies or --policy-selection")
    if args.policy_selection and args.config:
        parser.error("--policy-selection already contains a complete policy; omit --config")

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

    selected_policy = None
    selected_preset = args.preset
    if args.policy_selection:
        with args.policy_selection.open() as selection_file:
            selection = json.load(selection_file)
        selected_preset = selection.get("selected_preset")
        selected_policy = selection.get("selected_policy")
        if selected_preset not in BUILTIN_POLICIES or not isinstance(
            selected_policy, dict
        ):
            parser.error(f"Invalid policy selection file: {args.policy_selection}")
        validate_policy(selected_policy)
        print(f">> Applying saved global policy: {selected_preset}")
    elif args.evaluate_all_policies:
        selection = select_global_policy(
            patient_ids=patient_list,
            pred_dir=args.pred_dir,
            preprocessed_scan_pattern=args.preprocessed_scan_pattern,
            gt_scan_pattern=args.gt_scan_pattern,
            is_2d_input=args.is_2d_input,
            num_classes=args.num_classes,
            label_order=args.label_order,
        )
        selection_path = args.dest_folder / "postprocessing_policy.json"
        with selection_path.open("w") as selection_file:
            json.dump(selection, selection_file, indent=2)
        selected_preset = selection["selected_preset"]
        selected_policy = selection["selected_policy"]
        print(
            f">> Selected global policy {selected_preset} with mean patient/class "
            f"Dice {selection['selected_mean_patient_class_dice']:.4f}"
        )
        print(f">> Saved policy comparison to {selection_path}")

    pfun = partial(
        process_patient,
        pred_dir=args.pred_dir,
        preprocessed_scan_pattern=args.preprocessed_scan_pattern,
        raw_scan_pattern=args.raw_scan_pattern,
        gt_scan_pattern=args.gt_scan_pattern,
        output_dir=args.dest_folder,
        num_classes=args.num_classes,
        apply_post=args.post,
        is_2d_input=args.is_2d_input,
        preset=selected_preset,
        config_path=args.config,
        label_order=args.label_order,
        selected_policy=selected_policy,
    )

    if args.process == 1:
        for pid in tqdm(patient_list):
            pfun(pid)
    else:
        num_workers = None if args.process == -1 else args.process
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap(pfun, patient_list), total=len(patient_list)))

    print(f">> Pipeline complete! Saved volumes to {args.dest_folder}")


if __name__ == "__main__":
    main()