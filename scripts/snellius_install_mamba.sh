#!/bin/bash
#SBATCH --job-name=install
#SBATCH --output=scripts/slurm/install%j.log
#SBATCH --error=scripts/slurm/install%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0


cd $HOME/ai4mi_project
source ai4mi/bin/activate

echo "Downloading..."

wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_580.65.06_linux.run

echo "Extracting..."

sh cuda_13.0.0_580.65.06_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-temp

export PATH=$HOME/cuda-temp/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-temp/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$HOME/cuda-temp

nvcc --version

echo "Installing..."

export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export MAX_JOBS=16

pip install causal-conv1d>=1.4.0 --no-build-isolation

echo "Installed causal-conv1d"

MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm --no-build-isolation

echo "Installed mamba-ssm"

rm -rf $HOME/cuda-temp
rm cuda_13.0.0_580.65.06_linux.run

echo "Job Completed"
