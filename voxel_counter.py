from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd


def analyze_nested_segthor_stats(train_folder_path, label_names=None):
    train_dir = Path(train_folder_path)

    gt_files = sorted(list(train_dir.rglob("GT.nii.gz")))

    if not gt_files:
        print(f"No GT files found matching structure: {train_dir}/<patient_id>/GT.nii.gz")
        return None

    print(f"Found {len(gt_files)} patient GT files. Processing voxel counts...")

    records = []

    for gt_path in gt_files:
        patient_id = gt_path.parent.name

        img = nib.load(gt_path)
        data = img.get_fdata().astype(int)

        labels, counts = np.unique(data, return_counts=True)

        for label, count in zip(labels, counts):
            label_int = int(label)
            if label_int == 0:
                continue

            records.append(
                {
                    "Patient_ID": patient_id,
                    "Label": label_int,
                    "Voxel_Count": count,
                }
            )

    df = pd.DataFrame(records)

    if label_names:
        df["Organ"] = df["Label"].map(label_names)
        group_cols = ["Label", "Organ"]
    else:
        group_cols = ["Label"]

    stats_df = (
        df.groupby(group_cols)["Voxel_Count"]
        .agg(
            Present_In_Patients="count",
            Mean_Voxels="mean",
            Std_Dev="std",
            Min_Voxels="min",
            Median_Voxels="median",
            Max_Voxels="max",
        )
        .reset_index()
    )

    numeric_cols = [
        "Mean_Voxels",
        "Std_Dev",
        "Min_Voxels",
        "Median_Voxels",
        "Max_Voxels",
    ]
    stats_df[numeric_cols] = stats_df[numeric_cols].round(1)

    return stats_df, df


if __name__ == "__main__":
    TRAIN_DIR = "./data/segthor_part1/train"

    SEGTHOR_LABELS = {1: "Esophagus", 2: "Heart", 3: "Trachea", 4: "Aorta"}

    summary_table, per_patient_df = analyze_nested_segthor_stats(
        TRAIN_DIR, label_names=SEGTHOR_LABELS
    )

    if summary_table is not None:
        print("\n--- Dataset Voxel Count Statistics per Label ---")
        print(summary_table.to_string(index=False))

        # summary_table.to_csv("segthor_summary_stats.csv", index=False)
        # per_patient_df.to_csv("segthor_per_patient_counts.csv", index=False)
