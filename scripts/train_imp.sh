#!/bin/bash
#SBATCH --job-name=train_imp
#SBATCH --output=scripts/slurm/train_imp%j.log
#SBATCH --error=scripts/slurm/train_imp%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

cd $HOME/ai4mi_project
source ai4mi/bin/activate

echo "Data Preprocessing..."

# rm -rf data/SEGTHOR
# make data/SEGTHOR

MODEL="imp100"

echo "Training..."

python -O main.py \
    --model ImprovedENet \
    --loss compound \
    --use_focal \
    --batch_size 32 \
    --clip-grad 1.0 \
    --lr 0.001 \
    --epochs 100 \
    --warmup-epochs 3 \
    --dest results/segthor/$MODEL \
    --gpu \
    --compile \
    --augment

echo "Post Processing..."

python stitch.py --data_folder results/segthor/$MODEL/best_epoch/val \
    --dest_folder volumes/segthor/$MODEL \
    --num_classes 255 \
    --grp_regex "(Patient_\d\d)_\d\d\d\d" \
    --source_scan_pattern "data/segthor_train_full/train/{id_}/GT.nii.gz"

echo "Computing Metrics..."

python metrics.py \
    --data_folder results/segthor/$MODEL/best_epoch/val \
    --volumes_folder volumes/segthor/$MODEL \
    --target_pattern "data/segthor_train_full/train/{id_}/GT.nii.gz" \
    --grp_regex "(Patient_\d\d)_\d\d\d\d" \
    --num_classes 5 \
    --backend distorch \
    --device cuda

echo "Job Completed"
