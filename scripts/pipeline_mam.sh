#!/bin/bash
#SBATCH --job-name=pipeline_mam
#SBATCH --output=scripts/slurm/pipeline_mam%j.log
#SBATCH --error=scripts/slurm/pipeline_mam%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

cd $HOME/ai4mi_project
source ai4mi/bin/activate

export IS_SNELLIUS=1

echo "Data Preprocessing..."

rm -rf data/segthor_processed
python preprocess.py --input_dir data/segthor_train_full/train --output_dir data/segthor_processed/train -p 4

echo "Slicing..."

rm -rf data/SEGTHOR_processed
make data/SEGTHOR_processed

MODEL="MambaNet"
SEEDS=(42 43 44 45 46)

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Running pipeline for Seed: $SEED"
    echo "=========================================="

    # Define seed-specific destination folders
    DEST_RESULTS="results/segthor/${MODEL}/seed_${SEED}"
    DEST_VOLUMES="volumes/segthor/${MODEL}/seed_${SEED}"
    DEST_METRICS="volumes/segthor/${MODEL}/seed_${SEED}/results.csv"

    echo "Training (Seed $SEED)..."

    python -O main.py \
        --model MambaNet \
        --loss compound \
        --batch_size 32 \
        --clip-grad 1.0 \
        --lr 0.001 \
        --epochs 50 \
        --warmup-epochs 3 \
        --dataset SEGTHOR_processed \
        --dest "$DEST_RESULTS" \
        --seed "$SEED" \
        --deterministic \
        --gpu \
        --channels_last \
        --augment \
        --compile

    echo "Post Processing (Seed $SEED)..."

    python postprocess.py \
        --pred_dir "${DEST_RESULTS}/best_epoch/val" \
        --preprocessed_scan_pattern "data/segthor_processed/train/{id_}/GT.nii.gz" \
        --raw_scan_pattern "data/segthor_train_full/train/{id_}/GT.nii.gz" \
        --gt_scan_pattern "data/segthor_processed/train/{id_}/GT.nii.gz" \
        --dest_folder "$DEST_VOLUMES" \
        --grp_regex "^(Patient_\d+)" \
        --is_2d_input \
        --num_classes 5 \
        --post \
        --evaluate_all_policies \
        -p 5

    echo "Computing Metrics (Seed $SEED)..."

    python metrics.py \
        --volumes_folder "$DEST_VOLUMES" \
        --target_pattern "data/segthor_train_full/train/{id_}/GT.nii.gz" \
        --grp_regex "(Patient_\d\d)" \
        --num_classes 5 \
        --backend distorch \
        --device cuda \
        --csv_out "$DEST_METRICS"

done

echo "Job Completed"
