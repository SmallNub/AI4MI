#!/bin/bash
#SBATCH --job-name=segthor_pp
#SBATCH --output=scripts/slurm/segthor_pp_%j.out
#SBATCH --error=scripts/slurm/segthor_pp_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

# evaluate the post-processing policy grid on one (1) completed model run. 

# usage:
#   sbatch scripts/evaluate_postprocessing_policies.sh RUN_NAME RAW_NIFTI_FOLDER
#
# example for a 2D / 2.5D run after postprocess.py has reconstructed
# its slice predictions into original-space NIfTI volumes:
#   sbatch scripts/evaluate_postprocessing_policies.sh imp100 volumes/segthor/imp100
#
# example for a 3D run whose best_epoch folder already contains
# original-space NIfTI predictions:
#   sbatch scripts/evaluate_postprocessing_policies.sh imp3d50 results/segthor/imp3d50/best_epoch
#
# the input folder must contain files named Patient_XX.nii.gz that have the
# same shape, spacing and affine as their corresponding GT.nii.gz volumes.

set -euo pipefail

RUN_NAME="${1:?Usage: sbatch $0 RUN_NAME RAW_NIFTI_FOLDER}"
RAW_DIR="${2:?Usage: sbatch $0 RUN_NAME RAW_NIFTI_FOLDER}"

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$HOME/ai4mi/bin/activate"

cd "$HOME/medical/AI4MI"

GT_PATTERN="data/segthor_train_full/train/{id_}/GT.nii.gz"
ID_REGEX=".*(Patient_[0-9]{2}).*"
OUT_ROOT="volumes/segthor/${RUN_NAME}_literature_pp_v1"
METRICS_ROOT="results/segthor/metrics_${RUN_NAME}_literature_pp_v1"

PRESETS=(
  "none"
  "lcc_3d_non_esophagus"
  "esophagus_min_500mm3"
  "heart_3d_lcc_axial_2d_lcc"
  "relative_20pct_all"
)

if [[ ! -d "$RAW_DIR" ]]; then
    echo "ERROR: raw NIfTI prediction folder does not exist: $RAW_DIR"
    exit 1
fi

if ! find "$RAW_DIR" -type f -name "Patient_*.nii.gz" -print -quit | grep -q .; then
    echo "ERROR: no Patient_*.nii.gz files found below: $RAW_DIR"
    echo "Run postprocess.py reconstruction/resampling step first."
    exit 1
fi

mkdir -p scripts/slurm "$OUT_ROOT" "$METRICS_ROOT"

# verify geometry before spending GPU time on metrics.
python - "$RAW_DIR" <<'PY'
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

raw_dir = Path(sys.argv[1])
gt_root = Path("data/segthor_train_full/train")
paths = sorted(raw_dir.rglob("Patient_*.nii.gz"))

for prediction_path in paths:
    patient = prediction_path.name.removesuffix(".nii.gz")
    gt_path = gt_root / patient / "GT.nii.gz"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT for {patient}: {gt_path}")

    prediction = nib.load(prediction_path)
    target = nib.load(gt_path)

    if prediction.shape != target.shape:
        raise ValueError(
            f"{patient}: shape mismatch: {prediction.shape} vs {target.shape}"
        )
    if not np.allclose(prediction.header.get_zooms()[:3], target.header.get_zooms()[:3]):
        raise ValueError(
            f"{patient}: spacing mismatch: "
            f"{prediction.header.get_zooms()[:3]} vs {target.header.get_zooms()[:3]}"
        )
    if not np.allclose(prediction.affine, target.affine):
        raise ValueError(f"{patient}: affine mismatch")

    print(f"PASS geometry: {patient}")
PY

for PRESET in "${PRESETS[@]}"; do
    OUT_DIR="$OUT_ROOT/$PRESET"
    CSV_OUT="$METRICS_ROOT/$PRESET.csv"

    if [[ -e "$OUT_DIR" || -e "$CSV_OUT" ]]; then
        echo "ERROR: refusing to overwrite existing output for preset '$PRESET'"
        echo "Existing path: $OUT_DIR or $CSV_OUT"
        exit 1
    fi

    echo
    echo "="
    echo "RUN=$RUN_NAME | PRESET=$PRESET"
    echo "="

    python postprocessing.py \
        --input_folder "$RAW_DIR" \
        --output_folder "$OUT_DIR" \
        --preset "$PRESET" \
        --recursive

    python metrics.py \
        --data_folder "$OUT_DIR" \
        --volumes_folder "$OUT_DIR" \
        --target_pattern "$GT_PATTERN" \
        --grp_regex "$ID_REGEX" \
        --num_classes 5 \
        --backend distorch \
        --device cuda \
        --workers 1 \
        --csv_out "$CSV_OUT"
done

echo
printf 'completed policy evaluation for run: %s\n' "$RUN_NAME"
printf 'processed NIfTI folders: %s\n' "$OUT_ROOT"
printf 'metric CSVs: %s\n' "$METRICS_ROOT"
