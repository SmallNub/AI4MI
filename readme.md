# AI4MI project

This repository contains the segmentation pipeline used for the SegTHOR medical imaging project: prepare CT data, train a model, reconstruct 3D predictions, and evaluate the result with clinical-style volumetric metrics.

## Overview

The workflow in this project is:

1. preprocess raw CT/NIfTI volumes,
2. slice or convert them into training examples,
3. train a neural network in [main.py](main.py),
4. post-process the predicted masks back to 3D volumes,
5. score the results with [metrics.py](metrics.py),
6. aggregate per-seed results with [aggregate_results.py](aggregate_results.py).

The current setup is optimized for the SegTHOR task and for the Slurm-based scripts in [scripts/](scripts/). The canonical examples are [scripts/pipeline_ENet3D.sh](scripts/pipeline_ENet3D.sh) and [scripts/pipeline_mam.sh](scripts/pipeline_mam.sh): they train a model for multiple seeds, run post-processing, and compute final metrics for each run.

## Environment setup

This project expects a recent Python (3.12+). Create a virtual environment and install the base requirements:

```bash
python -m venv ai4mi
source ai4mi/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Important: the requirements file is not enough for the full pipeline.

The current workflow uses two packages that require special installation outside the normal requirements file:

- `distorch` is imported in [metrics.py](metrics.py) and is used as the default boundary metric backend.
- `mamba-ssm` is used by the Mamba-based models in [MambaNet.py](MambaNet.py) and [MambaNet3D.py](MambaNet3D.py).

Install the remaining special dependencies manually, because they are not in `requirements.txt`:

```bash
python -m pip install distorch
python -m pip install causal-conv1d>=1.4.0 --no-build-isolation
MAMBA_KEEP_CUDA_BUILD=TRUE python -m pip install mamba-ssm --no-build-isolation
```

The repo also includes a dedicated install script at [scripts/snellius_install_mamba.sh](scripts/snellius_install_mamba.sh), which does the same Mamba install with the CUDA build flags enabled.

This is why `requirements.txt` alone will not install the Mamba stack or the `distorch` metric backend.

## Main files and how they are used

- [main.py](main.py)
  - central training entry point.
  - defines the datasets, model registry, losses, optimizer, scheduler, and training loop.
  - supports commands like `--model`, `--dataset`, `--loss`, `--epochs`, `--gpu`, `--augment`, `--compile`, and seed control.

- [dataset.py](dataset.py)
  - contains the dataset classes used during training.
  - handles slice-based data (`SliceDataset`) and 3D CT data (`Segthor3DDataset`).
  - decides how data is transformed and split into train/validation.

- [preprocess.py](preprocess.py)
  - normalizes/prepares raw SegTHOR NIfTI volumes to a common spacing and intensity window.
  - used before slicing to create a consistent processed dataset.

- [slice_segthor.py](slice_segthor.py)
  - converts the CT + GT volumes into slice-based examples suitable for model training.
  - the repo’s `Makefile` wraps this in `make data/SEGTHOR` and `make data/SEGTHOR_processed`.

- [postprocess.py](postprocess.py)
  - reconstructs volume predictions from model outputs.
  - applies 3D cleanup policies (`keep largest component`, minimum voxel filters, slice-level cleanup, etc.).
  - writes processed predictions into a `volumes/...` folder for final evaluation.

- [metrics.py](metrics.py)
  - computes segmentation metrics such as Dice, HD95, ASSD, precision/recall, and relative volume error.
  - supports `--backend scipy` or `--backend distorch`; the Slurm scripts use `--backend distorch`.

- [aggregate_results.py](aggregate_results.py)
  - collects `results.csv` files across training seeds.
  - produces a model summary and boxplots for comparison.

- [viewer/viewer.py](viewer/viewer.py)
  - visual comparison tool for checking ground truth versus predictions slice-by-slice.

- [scripts/](scripts/)
  - contains the actual experiment workflow used in practice.
  - the scripts are the best source of truth for how this repo is run on the cluster.

## Full pipeline

The real pipeline for this project is the same pattern used in the Slurm scripts.

### 1) Prepare the raw data

```bash
python preprocess.py --input_dir data/segthor_train_full/train --output_dir data/segthor_processed/train -p 4
```

This prepares the raw SegTHOR CT volumes for a consistent spacing and intensity normalization.

Then slice the processed volumes into the training format used by the model:

```bash
make data/SEGTHOR_processed
```

or directly:

```bash
python slice_segthor.py --source_dir data/segthor_processed --dest_dir data/SEGTHOR_processed --shape 256 256 --retains 5 -p 1 --skip_preprocessing
```

### 2) Train the model

The project is typically run one seed at a time, then repeated for several seeds. The Slurm scripts show the intended pattern.

Example from [scripts/pipeline_ENet3D.sh](scripts/pipeline_ENet3D.sh):

```bash
python -O main.py \
  --model ENet3D_old \
  --mode full \
  --loss compound \
  --batch_size 1 \
  --clip-grad 1.0 \
  --lr 0.001 \
  --epochs 50 \
  --warmup-epochs 3 \
  --dataset segthor_processed \
  --dest results/ENet_base/dimensions/ENet3D/seed_42 \
  --seed 42 \
  --deterministic \
  --gpu \
  --channels_last \
  --augment \
  --compile
```

This writes the best model and validation predictions under the destination folder. The script loops across seeds like `42 43 44 45 46`.

### 3) Reconstruct 3D predictions and clean them up

After inference, the training folder is passed to [postprocess.py](postprocess.py):

```bash
python postprocess.py \
  --pred_dir "results/ENet_base/dimensions/ENet3D/seed_42/best_epoch" \
  --preprocessed_scan_pattern "data/segthor_processed/train/{id_}/GT.nii.gz" \
  --raw_scan_pattern "data/segthor_train_full/train/{id_}/GT.nii.gz" \
  --gt_scan_pattern "data/segthor_processed/train/{id_}/GT.nii.gz" \
  --dest_folder "volumes/ENet_base/dimensions/ENet3D/seed_42/val" \
  --grp_regex "^(Patient_\d+)" \
  --num_classes 5 \
  --post \
  --evaluate_all_policies \
  -p 5
```

This step takes the 2D predictions, reassembles patient volumes, and applies the configured post-processing policies before saving the final predictions.

### 4) Compute metrics

Once the reconstructed volumes are saved, evaluate them with [metrics.py](metrics.py):

```bash
python metrics.py \
  --volumes_folder "volumes/ENet_base/dimensions/ENet3D/seed_42/val" \
  --target_pattern "data/segthor_train_full/train/{id_}/GT.nii.gz" \
  --grp_regex "(Patient_\d\d)" \
  --num_classes 5 \
  --backend distorch \
  --device cuda \
  --csv_out "volumes/ENet_base/dimensions/ENet3D/seed_42/val/results.csv"
```

This produces a per-class summary table, usually with Dice, HD95, ASSD, and related metrics.

### 5) Aggregate across seeds

After running multiple seeds, collect all CSVs and compare models:

```bash
python aggregate_results.py results/
```

The script searches recursively for `results.csv` files, averages across seeds, and can generate a boxplot comparing models or seeds for a chosen metric.

## Typical runtime workflow

The project is currently run as a Slurm pipeline such as:

```bash
bash scripts/preprocess.sh
bash scripts/pipeline_ENet3D.sh
# or
bash scripts/pipeline_mam.sh
```

This is the current practical workflow reflected by the repo today: preprocessing → training → post-processing → metric computation → aggregation.

## Quick sanity checks

For a quick debug run without the full pipeline:

```bash
python main.py --dataset TOY2 --mode full --epochs 5 --dest results/toy2/debug --gpu --debug
```

This uses the synthetic toy dataset and is useful for checking that the training loop, logging, and output structure work before launching a full SegTHOR experiment.

## Viewer

To compare image and prediction slices visually:

```bash
python viewer/viewer.py \
  --img_source data/TOY2/val/img data/TOY2/val/gt \
  results/toy2/ce/iter000/val results/toy2/ce/iter005/val results/toy2/ce/best_epoch/val \
  --show_img -C 256 --no_contour
```

This is mainly useful for quick qualitative checks of model behavior.
