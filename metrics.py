#!/usr/bin/env python3

"""3D segmentation metrics for reconstructed NIfTI volumes across multiple patients."""

from __future__ import annotations

import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Match, Pattern

import torch
import distorch
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return mask ^ eroded


def _surface_distances(
    source: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]
) -> np.ndarray:
    target_surface = _surface(target)
    distances = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    return distances[_surface(source)]


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    pred_size = pred.sum()
    target_size = target.sum()

    if pred_size == 0 and target_size == 0:
        return 1.0

    return float(2 * np.logical_and(pred, target).sum() / (pred_size + target_size))


def precision(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    true_positive = np.logical_and(pred, target).sum()
    predicted_positive = pred.sum()

    if predicted_positive == 0:
        return 1.0 if target.sum() == 0 else 0.0

    return float(true_positive / predicted_positive)


def recall(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    true_positive = np.logical_and(pred, target).sum()
    target_positive = target.sum()

    if target_positive == 0:
        return 1.0 if pred.sum() == 0 else 0.0

    return float(true_positive / target_positive)


def relative_volume_error(pred: np.ndarray, target: np.ndarray) -> float:
    pred_volume = pred.astype(bool).sum()
    target_volume = target.astype(bool).sum()

    if target_volume == 0:
        return 0.0 if pred_volume == 0 else float("inf")

    return float(abs(pred_volume - target_volume) / target_volume)


def hd95(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> float:
    """Calculates Hausdorff distance at 95th percentile"""
    pred = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("inf")

    distances = np.concatenate(
        [
            _surface_distances(pred, target, spacing),
            _surface_distances(target, pred, spacing),
        ]
    )
    return float(np.percentile(distances, 95))


def assd(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> float:
    """Calculates the Average Symmetric Surface Distance"""
    pred = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("inf")

    distances = np.concatenate(
        [
            _surface_distances(pred, target, spacing),
            _surface_distances(target, pred, spacing),
        ]
    )
    return float(distances.mean())


def surface_dice(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...],
    tolerance: float = 1.0,
) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    pred_surface = _surface(pred)
    target_surface = _surface(target)

    if not pred_surface.any() and not target_surface.any():
        return 1.0
    if not pred_surface.any() or not target_surface.any():
        return 0.0

    pred_distances = _surface_distances(pred, target, spacing)
    target_distances = _surface_distances(target, pred, spacing)

    return float(
        ((pred_distances <= tolerance).sum() + (target_distances <= tolerance).sum())
        / (len(pred_distances) + len(target_distances))
    )


def metric_dict(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...],
    surface_tolerance: float = 1.0,
    backend: str = "scipy",
    device: str = "cuda",
) -> dict[str, float]:
    """Compute all metrics for one binary 3D class mask."""
    pred = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() and not target.any():
        hd95_value = assd_value = 0.0
        surface_dice_value = 1.0
    elif not pred.any() or not target.any():
        hd95_value = assd_value = float("inf")
        surface_dice_value = 0.0
    elif backend == "distorch":
        pred_tensor = (
            torch.from_numpy(np.ascontiguousarray(pred)).unsqueeze(0).to(device)
        )
        target_tensor = (
            torch.from_numpy(np.ascontiguousarray(target)).unsqueeze(0).to(device)
        )
        distance_metrics = distorch.boundary_metrics(
            pred_tensor,
            target_tensor,
            element_size=spacing,
            distance_threshold=surface_tolerance,
        )
        hd95_value = float(
            torch.maximum(
                distance_metrics.Hausdorff95_1_to_2,
                distance_metrics.Hausdorff95_2_to_1,
            ).item()
        )
        assd_value = float(distance_metrics.AverageSymmetricSurfaceDistance.item())
        surface_dice_value = float(
            distance_metrics.NormalizedSymmetricSurfaceDistance.item()
        )
    else:
        if backend != "scipy":
            raise ValueError(f"unknown metric backend: {backend}")
        pred_surface = _surface(pred)
        target_surface = _surface(target)

        if not pred_surface.any() and not target_surface.any():
            hd95_value = assd_value = 0.0
            surface_dice_value = 1.0
        elif not pred_surface.any() or not target_surface.any():
            hd95_value = assd_value = float("inf")
            surface_dice_value = 0.0
        else:
            target_distance_map = ndimage.distance_transform_edt(
                ~target_surface, sampling=spacing
            )
            pred_distance_map = ndimage.distance_transform_edt(
                ~pred_surface, sampling=spacing
            )
            pred_distances = target_distance_map[pred_surface]
            target_distances = pred_distance_map[target_surface]
            distances = np.concatenate([pred_distances, target_distances])
            hd95_value = float(np.percentile(distances, 95))
            assd_value = float(distances.mean())
            surface_dice_value = float(
                (
                    (pred_distances <= surface_tolerance).sum()
                    + (target_distances <= surface_tolerance).sum()
                )
                / len(distances)
            )

    return {
        "dice": dice(pred, target),
        "hd95_mm": hd95_value,
        "assd_mm": assd_value,
        "relative_volume_error": relative_volume_error(pred, target),
        "precision": precision(pred, target),
        "recall": recall(pred, target),
        "surface_dice": surface_dice_value,
    }


def volume_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...],
    num_classes: int,
    surface_tolerance: float = 1.0,
    workers: int = 1,
    progress: bool = False,
    backend: str = "scipy",
    device: str = "cuda",
) -> dict[int, dict[str, float]]:
    """Compute metrics for every foreground class in two label volumes."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
        )
    if len(spacing) != 3:
        raise ValueError("spacing must contain (x, y, z) voxel sizes")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if backend == "distorch" and workers != 1:
        raise ValueError("use workers=1 with the distorch backend")

    def compute_class(class_id: int) -> tuple[int, dict[str, float]]:
        return class_id, metric_dict(
            prediction == class_id,
            target == class_id,
            spacing,
            surface_tolerance,
            backend,
            device,
        )

    class_ids = list(range(1, num_classes))
    results: dict[int, dict[str, float]] = {}
    if workers == 1:
        iterator = (
            tqdm(class_ids, desc="Classes", unit="class", leave=False)
            if progress
            else class_ids
        )
        for class_id in iterator:
            result_class, result = compute_class(class_id)
            results[result_class] = result
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(compute_class, class_id) for class_id in class_ids]
        completed = as_completed(futures)
        if progress:
            completed = tqdm(
                completed, total=len(futures), desc="Classes", unit="class", leave=False
            )
        for future in completed:
            result_class, result = future.result()
            results[result_class] = result
    return results


def _load_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.load(str(path))
    return np.asarray(image.dataobj), tuple(
        float(value) for value in image.header.get_zooms()[:3]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch compute 3D segmentation metrics across patients"
    )

    # Can scan either PNG slice directory OR stitched volumes directory directly
    parser.add_argument(
        "--data_folder",
        type=Path,
        required=True,
        help="Path to 2D slices folder OR folder containing stitched NIfTI volumes",
    )
    parser.add_argument(
        "--volumes_folder",
        type=Path,
        default=None,
        help="Path to folder with stitched .nii.gz files. If omitted, uses data_folder.",
    )
    parser.add_argument(
        "--target_pattern",
        type=str,
        required=True,
        help="Pattern for target NIfTI files using {id_} placeholder (e.g. 'data/segthor_part1/train/{id_}/GT.nii.gz')",
    )
    parser.add_argument(
        "--grp_regex",
        type=str,
        required=True,
        help="Regex pattern with a matching group for patient ID",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of segmentation classes (including background)",
    )
    parser.add_argument("--surface_tolerance", type=float, default=1.0)
    parser.add_argument(
        "--backend",
        choices=["scipy", "distorch"],
        default="distorch",
        help="Boundary metric backend (default: distorch)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for DisTorch (default: cuda)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of classes to evaluate concurrently (default: 1)",
    )
    parser.add_argument(
        "--csv_out",
        type=Path,
        default=None,
        help="Optional path to save results as CSV",
    )
    args = parser.parse_args()

    v_folder = args.volumes_folder if args.volumes_folder else args.data_folder

    images: list[Path] = list(args.data_folder.glob("*.png"))
    grouping_regex: Pattern = re.compile(args.grp_regex)

    if images:
        stems: list[str] = [p.stem for p in images]
        matches: list[Match] = [grouping_regex.match(s) for s in stems if grouping_regex.match(s)]  # type: ignore
        unique_patients: list[str] = sorted(list({match.group(1) for match in matches}))
    else:
        # Fallback: scan for NIfTI volumes matching grp_regex directly
        nii_files = list(v_folder.glob("*.nii.gz"))
        unique_patients = []
        for p in nii_files:
            match = grouping_regex.match(p.name.replace(".nii.gz", ""))
            if match:
                unique_patients.append(match.group(1))
        unique_patients = sorted(list(set(unique_patients)))

    if not unique_patients:
        raise FileNotFoundError(
            f"No matching patients found in {args.data_folder} with regex '{args.grp_regex}'"
        )

    print(
        f"Found {len(unique_patients)} unique patients for evaluation: {unique_patients}"
    )

    records = []

    for patient_id in tqdm(unique_patients, desc="Evaluating Patients", unit="patient"):
        pred_path = (v_folder / patient_id).with_suffix(".nii.gz")
        target_path = Path(args.target_pattern.format(id_=patient_id))

        if not pred_path.exists():
            print(
                f"Warning: Prediction volume for {patient_id} at {pred_path} not found. Skipping..."
            )
            continue

        if not target_path.exists():
            print(
                f"Warning: Target ground truth for {patient_id} at {target_path} not found. Skipping..."
            )
            continue

        prediction, spacing = _load_volume(pred_path)
        target, target_spacing = _load_volume(target_path)

        if not np.allclose(spacing, target_spacing):
            raise ValueError(
                f"Spacing mismatch for patient {patient_id}: {spacing} vs {target_spacing}"
            )

        results = volume_metrics(
            prediction,
            target,
            spacing,
            args.num_classes,
            args.surface_tolerance,
            workers=args.workers,
            progress=False,
            backend=args.backend,
            device=args.device,
        )

        for class_id, metrics in results.items():
            record = {"patient": patient_id, "class": class_id}
            record.update(metrics)
            records.append(record)

    df = pd.DataFrame(records)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n" + "=" * 60)
    print(" PER-PATIENT / CLASS RESULTS ")
    print("=" * 60)
    print(df.to_string(index=False))

    summary = df.groupby("class").mean(numeric_only=True).reset_index()
    print("\n" + "=" * 60)
    print(" MEAN METRICS ACROSS ALL PATIENTS ")
    print("=" * 60)
    print(summary.to_string(index=False))

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)

        summary_rows = summary.copy()
        summary_rows.insert(0, "patient", "MEAN")

        combined = pd.concat(
            [df, summary_rows],
            ignore_index=True,
        )

        combined.to_csv(args.csv_out, index=False)
        print(f"\nDetailed and mean metrics saved to {args.csv_out}")


if __name__ == "__main__":
    main()
