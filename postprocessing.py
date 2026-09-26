#!/usr/bin/env python3
"""Start of pre-processing methods"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

LABELS = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}

# base policy: no changes. edit via --config after inspecting errors.
BASE_POLICY: dict[str, dict[str, Any]] = {
    str(label): {
        "enabled": False,
        "connectivity": 3,          # 1/2/3 gives 6/18/26-neighbour connectivity
        "keep_largest": False,
        "min_component_mm3": 0.0,   # no arbitrary voxel count thresh.
        "fill_holes": False,
        "closing_radius_mm": 0.0,   # zero disables closing.
    }
    for label in LABELS
}

# ablation
BUILTIN_POLICIES: dict[str, dict[str, dict[str, Any]]] = {
    "none": deepcopy(BASE_POLICY),
    "heart_lcc": {
        **deepcopy(BASE_POLICY),
        "2": {
            **deepcopy(BASE_POLICY["2"]),
            "enabled": True,
            "keep_largest": True,
        },
    },
}


def component_statistics(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int, np.ndarray]:
    """Return connected-component labels, count and component voxel counts."""
    structure = ndimage.generate_binary_structure(3, connectivity)
    labelled, n_components = ndimage.label(mask, structure=structure)
    counts = np.bincount(labelled.ravel(), minlength=n_components + 1)
    if len(counts):
        counts[0] = 0
    return labelled, n_components, counts


def ellipsoid(radius_mm: float, spacing: tuple[float, float, float]) -> np.ndarray:
    """Build a spacing-aware 3D structuring element for optional closing."""
    if radius_mm <= 0:
        return np.ones((1, 1, 1), dtype=bool)

    radii_vox = [max(1, int(np.ceil(radius_mm / axis_spacing))) for axis_spacing in spacing]
    grids = np.ogrid[
        -radii_vox[0] : radii_vox[0] + 1,
        -radii_vox[1] : radii_vox[1] + 1,
        -radii_vox[2] : radii_vox[2] + 1,
    ]
    normalized_distance = sum(
        (grid * spacing[axis] / radius_mm) ** 2
        for axis, grid in enumerate(grids)
    )
    return normalized_distance <= 1.0


def apply_policy(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    policy: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply a single class policy to a binary mask and return an audit record."""
    connectivity = int(policy.get("connectivity", 3))
    labelled, n_before, counts_before = component_statistics(mask, connectivity)
    voxel_volume_mm3 = float(np.prod(spacing))

    report = {
        "components_before": float(n_before),
        "largest_component_mm3_before": float(counts_before.max() * voxel_volume_mm3)
        if n_before else 0.0,
        "input_voxels": float(mask.sum()),
        "removed_voxels": 0.0,
        "added_voxels": 0.0,
    }

    if not policy.get("enabled", False):
        report["components_after"] = float(n_before)
        report["output_voxels"] = float(mask.sum())
        return mask.copy(), report

    keep_ids = np.arange(1, n_before + 1)
    if n_before and bool(policy.get("keep_largest", False)):
        keep_ids = np.array([int(np.argmax(counts_before))])

    min_component_mm3 = float(policy.get("min_component_mm3", 0.0))
    if n_before and min_component_mm3 > 0:
        sufficiently_large = np.where(counts_before * voxel_volume_mm3 >= min_component_mm3)[0]
        sufficiently_large = sufficiently_large[sufficiently_large != 0]
        keep_ids = np.intersect1d(keep_ids, sufficiently_large)

    cleaned = np.isin(labelled, keep_ids)

    # hole filling and closing are opt-in. can change topology >
    # so use only after qualitative inspection
    if bool(policy.get("fill_holes", False)):
        cleaned = ndimage.binary_fill_holes(cleaned)

    closing_radius_mm = float(policy.get("closing_radius_mm", 0.0))
    if closing_radius_mm > 0:
        cleaned = ndimage.binary_closing(
            cleaned,
            structure=ellipsoid(closing_radius_mm, spacing),
        )

    _, n_after, _ = component_statistics(cleaned, connectivity)
    report["components_after"] = float(n_after)
    report["output_voxels"] = float(cleaned.sum())
    report["removed_voxels"] = float(np.logical_and(mask, ~cleaned).sum())
    report["added_voxels"] = float(np.logical_and(~mask, cleaned).sum())
    return cleaned.astype(bool), report


def postprocess_volume(
    labels: np.ndarray,
    spacing: tuple[float, float, float],
    policy_by_label: dict[str, dict[str, Any]],
    label_order: list[int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """apply policies to an exclusive multiclass label map.

    morphological additions may be written only into original background. avoids
    overwriting a different predicted organ merely because classes are processed in
    sequence.
    """
    if labels.ndim != 3:
        raise ValueError(f"Expected one 3D label map, got shape {labels.shape}")

    unique = set(np.unique(labels).astype(int))
    allowed = {0, *LABELS}
    if not unique.issubset(allowed):
        raise ValueError(f"Expected labels {sorted(allowed)}, found {sorted(unique)}")

    source = labels.astype(np.uint8, copy=True)
    output = source.copy()
    records: list[dict[str, Any]] = []

    for class_id in label_order:
        if class_id not in LABELS:
            raise ValueError(f"Unknown class ID {class_id}")

        original_mask = source == class_id
        cleaned_mask, report = apply_policy(
            original_mask,
            spacing,
            policy_by_label[str(class_id)],
        )

        # remove original class > then restore the policy output. new voxel may
        # occupy background only > existing other-organ predictions remain untouched.
        output[original_mask] = 0
        writable = (source == 0) | original_mask
        output[cleaned_mask & writable & (output == 0)] = class_id

        records.append(
            {
                "class": class_id,
                "organ": LABELS[class_id],
                "enabled": bool(policy_by_label[str(class_id)].get("enabled", False)),
                **report,
            }
        )

    return output, records


def load_policy(preset: str, config_path: Path | None) -> dict[str, dict[str, Any]]:
    if preset not in BUILTIN_POLICIES:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(BUILTIN_POLICIES)}")

    policy = deepcopy(BUILTIN_POLICIES[preset])
    if config_path is None:
        return policy

    with config_path.open() as file:
        supplied = json.load(file)
    for label, overrides in supplied.items():
        if label not in policy:
            raise ValueError(f"Unknown label {label!r}; use strings '1' through '4'")
        if not isinstance(overrides, dict):
            raise ValueError(f"Policy for label {label} must be a JSON object")
        policy[label].update(overrides)
    return policy


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(BASE_POLICY, indent=2) + "\n")
    print(f"Wrote policy template: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="apply auditable per-organ 3D post-processing to SegTHOR NIfTI predictions."
    )
    parser.add_argument("--input_folder", type=Path, help="Folder containing predicted .nii.gz volumes")
    parser.add_argument("--output_folder", type=Path, help="New folder for processed .nii.gz volumes")
    parser.add_argument(
        "--preset",
        choices=sorted(BUILTIN_POLICIES),
        default="none",
        help="Small built-in policy set. Default 'none' copies predictions unchanged.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON overrides per class")
    parser.add_argument("--label_order", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--write_template", type=Path, default=None, help="Write a conservative JSON template and exit")
    args = parser.parse_args()

    if args.write_template is not None:
        write_template(args.write_template)
        return

    if args.input_folder is None or args.output_folder is None:
        parser.error("--input_folder and --output_folder are required unless --write_template is used")
    if not args.input_folder.is_dir():
        raise FileNotFoundError(args.input_folder)

    volumes = sorted(args.input_folder.glob("*.nii.gz"))
    if not volumes:
        raise FileNotFoundError(f"No .nii.gz volumes found in {args.input_folder}")

    policy = load_policy(args.preset, args.config)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    print("Effective policy:\n" + json.dumps(policy, indent=2))

    report_rows: list[dict[str, Any]] = []
    for input_path in volumes:
        image = nib.load(str(input_path))
        labels = np.asarray(image.dataobj)
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        processed, records = postprocess_volume(labels, spacing, policy, args.label_order)

        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        result = nib.Nifti1Image(processed.astype(np.uint8), image.affine, header)
        output_path = args.output_folder / input_path.name
        nib.save(result, str(output_path))

        for record in records:
            record.update(
                {
                    "patient": input_path.name.removesuffix(".nii.gz"),
                    "changed_voxels_total": int(np.count_nonzero(processed != labels)),
                }
            )
            report_rows.append(record)

        print(f"{input_path.name}: changed {np.count_nonzero(processed != labels)} voxels")

    report_path = args.output_folder / "postprocessing_report.csv"
    with report_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Wrote audit report: {report_path}")


if __name__ == "__main__":
    main()