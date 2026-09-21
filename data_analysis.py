#!/usr/bin/env python3
"""
Data analysis

"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import nibabel as nib
import numpy as np


def inspect_dataset(data_root: Path) -> None:
    train_root = data_root / "train"
    patients = sorted(train_root.glob("Patient_*"))

    if not patients:
        raise FileNotFoundError(f"no Patient_* folders found in {train_root}")

    label_counter: Counter[int] = Counter()
    rows: list[tuple[str, tuple[int, int, int], tuple[float, float, float], list[int]]] = []

    for patient_dir in patients:
        gt_path = patient_dir / "GT.nii.gz"
        ct_path = patient_dir / f"{patient_dir.name}.nii.gz"

        if not gt_path.exists():
            print(f"missing GT: {gt_path}")
            continue
        if not ct_path.exists():
            print(f"missing CT: {ct_path}")
            continue

        gt_nii = nib.load(str(gt_path))
        ct_nii = nib.load(str(ct_path))
        gt = np.asarray(gt_nii.dataobj)
        ct = np.asarray(ct_nii.dataobj)

        if gt.shape != ct.shape:
            print(f"shape mismatch for {patient_dir.name}: GT {gt.shape}, CT {ct.shape}")

        values, counts = np.unique(gt, return_counts=True)
        label_counter.update({int(v): int(c) for v, c in zip(values, counts)})
        spacing = tuple(float(x) for x in ct_nii.header.get_zooms()[:3])
        rows.append((patient_dir.name, tuple(int(x) for x in ct.shape), spacing, [int(v) for v in values]))

    print("\ndata audit")
    print(f"patients w/ readable CT and GT: {len(rows)}")
    print(f"patient IDs: {[row[0] for row in rows]}")

    print("\nper-patient geometry & observed labels:")
    for patient, shape, spacing, labels in rows:
        print(f"{patient:10s} shape={shape} spacing_mm={spacing} labels={labels}")

    shapes = np.asarray([row[1] for row in rows], dtype=float)
    spacings = np.asarray([row[2] for row in rows], dtype=float)
    print("\nobserved geometry summary:")
    print(f"shape min: {tuple(shapes.min(axis=0).astype(int))}")
    print(f"shape max: {tuple(shapes.max(axis=0).astype(int))}")
    print(f"spacing min (mm): {tuple(np.round(spacings.min(axis=0), 4))}")
    print(f"spacing max (mm): {tuple(np.round(spacings.max(axis=0), 4))}")

    total = sum(label_counter.values())
    print("\nlabel voxel counts across all gt volumes:")
    for label in sorted(label_counter):
        count = label_counter[label]
        print(f"label={label}: {count:,} voxels ({100 * count / total:.4f}%)")

    print("\nforeground presence by patient:")
    for label in sorted(k for k in label_counter if k != 0):
        present = sum(label in row[3] for row in rows)
        print(f"label={label}: present in {present}/{len(rows)} patients")


def inspect_results(results_root: Path) -> None:
    print("\nresult")
    if not results_root.exists():
        print(f"results directory does not exist: {results_root}")
        return

    wanted = [
        "best_epoch.txt",
        "metrics.csv",
        "bestweights.pt",
        "bestmodel.pkl",
        "dice_val.npy",
        "loss_val.npy",
    ]

    candidates = sorted([p for p in results_root.rglob("*") if p.is_file()])
    if not candidates:
        print(f"no files found under {results_root}")
        return

    for name in wanted:
        found = [p for p in candidates if p.name == name]
        if found:
            print(f"\n{name}:")
            for path in found:
                print(f"  {path}")
                if name in {"best_epoch.txt"}:
                    print("  content:")
                    print("  " + path.read_text(errors="replace").replace("\n", "\n  ").rstrip())

    print("\nall CSV files:")
    for path in [p for p in candidates if p.suffix == ".csv"]:
        print(f"  {path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="dir containing train/Patient_XX/{Patient_XX.nii.gz,GT.nii.gz}",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="optional results directory to scan for saved metadata and metrics",
    )
    args = parser.parse_args()

    inspect_dataset(args.data_root)
    if args.results_root is not None:
        inspect_results(args.results_root)


if __name__ == "__main__":
    main()
