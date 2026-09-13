#!/usr/bin/env python3

"""3D segmentation metrics for reconstructed NIfTI volumes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import torch
import distorch

import nibabel as nib
import numpy as np
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
        raise ValueError("prediction and target must be matching 3D arrays")
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
            tqdm(class_ids, desc="Computing metrics", unit="class")
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
                completed, total=len(futures), desc="Computing metrics", unit="class"
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
    parser = argparse.ArgumentParser(description="Compute 3D segmentation metrics")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--surface-tolerance", type=float, default=1.0)
    parser.add_argument(
        "--backend",
        choices=["scipy", "distorch"],
        default="scipy",
        help="Boundary metric backend (default: scipy)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for DisTorch, for example cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of classes to evaluate concurrently (default: 1)",
    )
    args = parser.parse_args()

    prediction, spacing = _load_volume(args.prediction)
    target, target_spacing = _load_volume(args.target)
    if not np.allclose(spacing, target_spacing):
        raise ValueError(f"spacing mismatch: {spacing} vs {target_spacing}")

    results = volume_metrics(
        prediction,
        target,
        spacing,
        args.num_classes,
        args.surface_tolerance,
        workers=args.workers,
        progress=True,
        backend=args.backend,
        device=args.device,
    )

    for class_id, metrics in results.items():
        formatted = ", ".join(f"{name}={value:.5g}" for name, value in metrics.items())
        print(f"class {class_id}: {formatted}")


if __name__ == "__main__":
    main()
