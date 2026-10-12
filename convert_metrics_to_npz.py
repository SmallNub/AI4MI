#!/usr/bin/env python3
"""Convert long-form patient/class metric CSVs to Distorch-style NPZ archives."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


SUMMARY_PATIENTS = {"mean", "average", "summary"}


def convert_metrics(csv_path: Path, output_dir: Path) -> list[Path]:
    """Write one NPZ per metric, mapping each patient ID to its class values."""
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"patient", "class"}
        columns = set(reader.fieldnames or ())
        missing = required - columns
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        metric_names = [name for name in (reader.fieldnames or []) if name not in required]
        if not metric_names:
            raise ValueError("CSV contains no metric columns")

        values: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
        class_ids: set[int] = set()
        for row in reader:
            patient_id = (row.get("patient") or "").strip()
            if not patient_id or patient_id.lower() in SUMMARY_PATIENTS:
                continue
            try:
                class_id = int(row["class"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid class ID for patient {patient_id!r}: {row.get('class')!r}"
                ) from error
            if class_id in values[patient_id]:
                raise ValueError(
                    f"Duplicate row for patient {patient_id!r}, class {class_id}"
                )

            class_values: dict[str, float] = {}
            for metric_name in metric_names:
                raw_value = row.get(metric_name)
                try:
                    class_values[metric_name] = float(raw_value)  # supports inf
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid {metric_name!r} value for patient {patient_id!r}, "
                        f"class {class_id}: {raw_value!r}"
                    ) from error

            values[patient_id][class_id] = class_values
            class_ids.add(class_id)

    if not values:
        raise ValueError("CSV contains no patient metric rows")

    ordered_classes = sorted(class_ids)
    for patient_id, patient_values in values.items():
        if set(patient_values) != class_ids:
            missing_classes = sorted(class_ids - set(patient_values))
            raise ValueError(
                f"Patient {patient_id!r} is missing class rows: {missing_classes}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric_name in metric_names:
        archive = {
            patient_id: np.asarray(
                [patient_values[class_id][metric_name] for class_id in ordered_classes],
                dtype=np.float64,
            )
            for patient_id, patient_values in sorted(values.items())
        }
        output_path = output_dir / f"{metric_name}.npz"
        np.savez_compressed(output_path, **archive)
        written.append(output_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a patient/class metric CSV to one NPZ archive per metric. "
            "Each archive maps patient IDs to class-ordered 1D arrays."
        )
    )
    parser.add_argument("csv_file", type=Path, help="Input metrics CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <csv-parent>/metrics_npz)",
    )
    args = parser.parse_args()

    if not args.csv_file.is_file():
        parser.error(f"CSV file not found: {args.csv_file}")
    output_dir = args.output_dir or args.csv_file.parent / "metrics_npz"
    paths = convert_metrics(args.csv_file, output_dir)
    print(f"Converted metrics for {args.csv_file}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()