#!/bin/bash
#SBATCH --job-name=train_vit
#SBATCH --output=scripts/slurm/train_vit%j.log
#SBATCH --error=scripts/slurm/train_vit%j.err
#SBATCH --time=4:00:00
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

MODEL="ViT"

echo "Training..."

python -O main.py \
    --model ViT \
    --loss compound \
    --use_focal \
    --batch_size 32 \
    --clip-grad 1.0 \
    --lr 0.001 \
    --epochs 50 \
    --warmup-epochs 3 \
    --dataset SEGTHOR_processed \
    --dest results/segthor/$MODEL \
    --gpu \
    --channels_last \
    --augment \
    --compile

echo "Post Processing..."

python postprocess.py \
    --pred_dir results/segthor/$MODEL/best_epoch/val \
    --preprocessed_scan_pattern "data/segthor_processed/train/{id_}/GT.nii.gz" \
    --raw_scan_pattern "data/segthor_train_full/train/{id_}/GT.nii.gz" \
    --dest_folder volumes/segthor/$MODEL \
    --grp_regex "^(Patient_\d+)" \
    --is_2d_input \
    --num_classes 5 \
    -p 4

echo "Computing Metrics..."

python metrics.py \
    --volumes_folder volumes/segthor/$MODEL \
    --target_pattern "data/segthor_train_full/train/{id_}/GT.nii.gz" \
    --grp_regex "(Patient_\d\d)" \
    --num_classes 5 \
    --backend distorch \
    --device cuda

echo "Job Completed"
