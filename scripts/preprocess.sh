#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=scripts/slurm/preprocess%j.log
#SBATCH --error=scripts/slurm/preprocess%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=16

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

echo "Job Completed"
