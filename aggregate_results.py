#!/usr/bin/env python3

"""Aggregate metrics/results.csv files across seeds and plot model scores."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED_PATTERN = re.compile(r"^seed[_-]?(?P<seed>.+)$", re.IGNORECASE)
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}
CLASS_NAMES = {
    1: "Esophagus",
    2: "Heart",
    3: "Trachea",
    4: "Aorta",
}


def class_name(class_id: float) -> str:
    return CLASS_NAMES.get(int(class_id), f"Class {class_id:g}")


def identify_model_and_seed(csv_path: Path, root: Path) -> tuple[str, str]:
    """Infer a model path and seed label from a results.csv location."""
    relative_parts = csv_path.parent.relative_to(root).parts
    seed_index = next(
        (
            index
            for index, part in enumerate(relative_parts)
            if SEED_PATTERN.match(part)
        ),
        None,
    )

    if seed_index is not None:
        model_parts = relative_parts[:seed_index]
        seed_match = SEED_PATTERN.match(relative_parts[seed_index])
        assert seed_match is not None
        seed = seed_match.group("seed")
    else:
        model_parts = tuple(
            part for part in relative_parts if part.lower() not in SPLIT_NAMES
        )
        seed = "/".join(relative_parts) or csv_path.parent.name

    model = "/".join(model_parts) or root.name
    return model, seed


def read_seed_metrics(csv_path: Path) -> pd.DataFrame:
    """Average patient metrics within each class for one seed."""
    frame = pd.read_csv(csv_path)
    if "class" not in frame.columns:
        raise ValueError(f"Missing required 'class' column in {csv_path}")
    if "patient" in frame.columns:
        frame = frame[frame["patient"].astype(str).str.upper() != "MEAN"]

    metric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column.lower() not in {"class", "seed"}
    ]
    if not metric_columns:
        raise ValueError(f"No numeric metric columns found in {csv_path}")

    for column in ["class", *metric_columns]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["class"])
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return (
        frame.groupby("class", as_index=False)[metric_columns]
        .mean()
        .melt(id_vars="class", var_name="metric", value_name="value")
        .dropna(subset=["value"])
    )


def collect_seed_metrics(root: Path) -> pd.DataFrame:
    records: list[dict[str, str | float]] = []
    csv_paths = sorted(root.rglob("results.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No results.csv files found under {root}")

    for csv_path in csv_paths:
        model, seed = identify_model_and_seed(csv_path, root)
        try:
            metrics = read_seed_metrics(csv_path)
        except (OSError, pd.errors.ParserError, ValueError) as error:
            raise ValueError(
                f"Could not read metrics from {csv_path}: {error}"
            ) from error

        for row in metrics.to_dict(orient="records"):
            records.append(
                {
                    "model": model,
                    "seed": seed,
                    "source": str(csv_path),
                    "class": row["class"],
                    "metric": row["metric"],
                    "value": row["value"],
                }
            )

    return pd.DataFrame.from_records(records)


def summarize(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = seed_metrics.groupby(["model", "class", "metric"], as_index=False)[
        "value"
    ].agg(mean="mean", std="std", seed_count="count")
    summary["class"] = summary["class"].map(class_name)
    return summary[["model", "class", "metric", "seed_count", "mean", "std"]]


def parse_model_labels(label_specs: list[str] | None) -> dict[str, str]:
    labels: dict[str, str] = {}
    for spec in label_specs or []:
        model_id, separator, label = spec.partition("=")
        if not separator or not model_id.strip() or not label.strip():
            raise ValueError(
                f"Invalid model label {spec!r}; expected MODEL_ID=DISPLAY_LABEL"
            )
        labels[model_id.strip()] = label.strip()
    return labels


def save_boxplot(
    seed_metrics: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str | None = None,
    class_labels: list[str] | None = None,
    model_labels: dict[str, str] | None = None,
) -> None:
    selected = seed_metrics[seed_metrics["metric"] == metric]
    if selected.empty:
        available = ", ".join(sorted(seed_metrics["metric"].unique()))
        raise ValueError(f"Metric {metric!r} not found. Available metrics: {available}")

    model_names = sorted(selected["model"].unique())
    model_display_names = [
        (model_labels or {}).get(model, model) for model in model_names
    ]
    classes = sorted(selected["class"].unique())
    if class_labels is not None and len(class_labels) != len(classes):
        raise ValueError(
            f"Received {len(class_labels)} class labels for {len(classes)} classes "
            f"({', '.join(class_name(class_id) for class_id in classes)})"
        )
    panel_labels = (
        class_labels
        if class_labels is not None
        else [class_name(class_id) for class_id in classes]
    )

    fig, axes = plt.subplots(
        1,
        len(classes),
        figsize=(max(8, len(model_names) * 1.2 * len(classes)), 5.5),
        squeeze=False,
    )
    for class_id, panel_label, ax in zip(classes, panel_labels, axes[0]):
        class_data = selected[selected["class"] == class_id]
        model_values = [
            class_data.loc[class_data["model"] == model, "value"].to_numpy()
            for model in model_names
        ]
        ax.boxplot(model_values, tick_labels=model_display_names, showmeans=True)
        ax.set_title(panel_label)
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find results.csv files recursively, aggregate metrics across seeds, "
            "and create a box plot comparing models."
        )
    )
    parser.add_argument("folder", type=Path, help="Folder to search recursively")
    parser.add_argument(
        "--metric",
        default="dice",
        help="Metric for the box plot (default: dice); all numeric metrics are summarized",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional overall title for the box plot",
    )
    parser.add_argument(
        "--class-labels",
        "--plot-labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help=(
            "Optional panel titles in ascending class-ID order; provide one per "
            "class present in the selected metric"
        ),
    )
    parser.add_argument(
        "--model-labels",
        nargs="+",
        default=None,
        metavar="MODEL_ID=LABEL",
        help=(
            "Optional model display-name mappings, e.g. "
            "compound=Improved_ENet ce=Baseline"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: input folder)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {root}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else root
    seed_metrics = collect_seed_metrics(root)
    summary = summarize(seed_metrics)
    model_labels = parse_model_labels(args.model_labels)
    if model_labels:
        summary["model"] = summary["model"].replace(model_labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(output_dir / "seed_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    plot_path = output_dir / f"boxplot_{args.metric}.png"
    save_boxplot(
        seed_metrics,
        args.metric,
        plot_path,
        title=args.title,
        class_labels=args.class_labels,
        model_labels=model_labels,
    )

    print(summary.to_string(index=False))
    print(f"Seed-level metrics: {output_dir / 'seed_metrics.csv'}")
    print(f"Mean/std summary: {output_dir / 'summary.csv'}")
    print(f"Box plot: {plot_path}")


if __name__ == "__main__":
    main()
