#!/usr/bin/env python3
"""Removal-only 3D post-processing for SegTHOR NIfTI predictions.

Implemented candidate policies are restricted to operations directly motivated by
SegTHOR2019 papers (see postprocessing.md).

"""

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
POLICY_FIELDS = {
    "enabled",
    "connectivity",
    "keep_largest",
    "min_component_mm3",
    "min_relative_to_largest",
    "slice_lcc_axis",
    "slice_connectivity",
}

# base policy: no changes. edit via --config after inspecting errors.
BASE_POLICY: dict[str, dict[str, Any]] = {
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
    policy: dict[str, dict[str, Any]], **overrides: Any
) -> dict[str, dict[str, Any]]:
    """return a copied policy with the same field overrides for selected labels."""
    copied = deepcopy(policy)
    for label, values in overrides.items():
        copied[label].update(values)
    return copied

# every policy below maps to a specific literature-motivated candidate (see postprocessing.md).
BUILTIN_POLICIES: dict[str, dict[str, dict[str, Any]]] = {
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
) -> tuple[np.ndarray, int, np.ndarray]:
    """label 3D components and return labels, count and voxel counts.
    """
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
    """retain largest 2D component independently in each slice.
    """
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

def validate_policy(policy_by_label: dict[str, dict[str, Any]]) -> None:
    """reject malformed policy values before any prediction volume is modified."""
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
    spacing: tuple[float, float, float],
    policy: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """apply one policy to a class mask.

    resulting mask is always a subset of the input mask. this prevents added
    voxels and prevents any possible overwrite of another organ in a multiclass map.
    """
    connectivity = int(policy["connectivity"])
    labelled, n_before, counts_before = component_statistics(mask, connectivity)
    voxel_volume_mm3 = float(np.prod(spacing))
    input_voxels = int(mask.sum())

    report: dict[str, Any] = {
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
    spacing: tuple[float, float, float],
    policy_by_label: dict[str, dict[str, Any]],
    label_order: list[int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """apply per-class policies to a 3D label map.
    """
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
    records: list[dict[str, Any]] = []

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

def load_policy(preset: str, config_path: Path | None) -> dict[str, dict[str, Any]]:
    """
    """
    if preset not in BUILTIN_POLICIES:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(BUILTIN_POLICIES)}")

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


def write_template(path: Path) -> None:
    """"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(BASE_POLICY, indent=2) + "\n")
    print(f"Wrote policy template: {path}")


def main() -> None:
    """"""
    parser = argparse.ArgumentParser(
        description=(
            "Apply 3D post-processing "
            "to SegTHOR NIfTI predictions."
        )
    )
    parser.add_argument(
        "--input_folder",
        type=Path,
        help="Folder containing predicted .nii.gz volumes",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Recursively discover .nii.gz predictions below --input_folder. "
            "Use this for nested layouts such as "
            "INPUT/Patient_01/Patient_01.nii.gz."
        ),
    )
    parser.add_argument(
        "--output_folder", type=Path, help="New folder for processed .nii.gz volumes"
    )
    parser.add_argument(
        "--preset",
        choices=sorted(BUILTIN_POLICIES),
        default="none",
        help="Named policy. Default 'none' changes no voxels.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON overrides for a named preset.",
    )
    parser.add_argument(
        "--label_order",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Class processing order; must contain labels 1, 2, 3 and 4 once each.",
    )
    parser.add_argument(
        "--write_template",
        type=Path,
        default=None,
        help="Write a no-op JSON policy template and exit.",
    )
    args = parser.parse_args()

    if args.write_template is not None:
        write_template(args.write_template)
        return

    if args.input_folder is None or args.output_folder is None:
        parser.error("--input_folder and --output_folder are required unless --write_template is used")
    if not args.input_folder.is_dir():
        raise FileNotFoundError(args.input_folder)

    search = args.input_folder.rglob if args.recursive else args.input_folder.glob
    volumes = sorted(
        path
        for path in search("*.nii.gz")
        if path.is_file()
    )

    if not volumes:
        mode = "recursively below" if args.recursive else "directly inside"
        raise FileNotFoundError(
            f"No .nii.gz volumes found {mode} {args.input_folder}"
        )

    names = [path.name for path in volumes]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(
            "Recursive input contains duplicate NIfTI filenames, which would "
            f"collide in the flat output folder: {duplicate_names}"
        )

    policy = load_policy(args.preset, args.config)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    print("Effective policy:\n" + json.dumps(policy, indent=2))

    report_rows: list[dict[str, Any]] = []
    for input_path in volumes:
        image = nib.load(str(input_path))
        labels = np.asarray(image.dataobj)
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        processed, records = postprocess_volume(
            labels, spacing, policy, args.label_order
        )

        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        result = nib.Nifti1Image(processed.astype(np.uint8), image.affine, header)
        output_path = args.output_folder / input_path.name
        nib.save(result, str(output_path))

        changed_voxels_total = int(np.count_nonzero(processed != labels))
        for record in records:
            record.update(
                {
                    "patient": input_path.name.removesuffix(".nii.gz"),
                    "preset": args.preset,
                    "changed_voxels_total": changed_voxels_total,
                }
            )
            report_rows.append(record)

        print(f"{input_path.name}: changed {changed_voxels_total} voxels")

    report_path = args.output_folder / "postprocessing_report.csv"
    with report_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Wrote audit report: {report_path}")

if __name__ == "__main__":
    main()

