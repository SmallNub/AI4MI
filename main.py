#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import gc
import os
import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from shutil import copytree, rmtree
from functools import partial

import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

from skimage.transform import resize
from torchinfo import summary

from dataset import SliceDataset, Segthor3DDataset
from ShallowNet import shallowCNN
from ENet import ENet, AttentionENet, SpatialENet, CBAMENet, LateFusionENet
from MambaNet import MambaNet
from MambaNet3D import MambaNet3D
from ImprovedENet import ImprovedENet
from ImprovedENet3D import ImprovedENet3D
from ENet3D import ENet3D, AttentionENet3D
from ViT import ViT
from utils import (
    Dcm,
    class2one_hot,
    probs2one_hot,
    probs2class,
    tqdm_,
    dice_coef,
    save_images,
    seed_everything,
    seed_worker,
)

from losses import CrossEntropy, FocalLoss, GeneralizedDice, CompoundLoss, TverskyLoss

datasets_params: dict[str, dict[str, Any]] = {}
datasets_params["TOY2"] = {"K": 2, "B": 2}
datasets_params["SEGTHOR"] = {"K": 5, "B": 8}
datasets_params["SEGTHOR_processed"] = {"K": 5, "B": 8}
datasets_params["segthor_train_full"] = {"K": 5, "B": 4}
datasets_params["segthor_processed"] = {"K": 5, "B": 4}

models_params: dict[str, dict[str, Any]] = {}
models_params["shallowCNN"] = {"net": shallowCNN, "args": {"kernels": 8, "factor": 2}}
models_params["ENet"] = {"net": ENet, "args": {"kernels": 8, "factor": 2}}
models_params["AttentionENet"] = {
    "net": AttentionENet,
    "args": {"kernels": 8, "factor": 2},
}
models_params["SpatialENet"] = {
    "net": SpatialENet,
    "args": {"kernels": 8, "factor": 2},
}
models_params["CBAMENet"] = {
    "net": CBAMENet,
    "args": {"kernels": 8, "factor": 2},
}
models_params["LateFusionENet"] = {
    "net": LateFusionENet,
    "args": {"kernels": 8, "factor": 2, "z_window": 5},
}
models_params["ImprovedENet"] = {
    "net": ImprovedENet,
    "args": {"kernels": 8, "z_window": 15},
}
models_params["ViT"] = {
    "net": ViT,
    "args": {"kernels": 16, "z_window": 15},
}
models_params["ImprovedENet3D"] = {
    "net": ImprovedENet3D,
    "args": {"kernels": 16},
}
models_params["MambaNet"] = {
    "net": MambaNet,
    "args": {"kernels": 16, "z_window": 15},
}
models_params["MambaNet3D"] = {
    "net": MambaNet3D,
    "args": {"kernels": 2},
}
models_params["ENet3D"] = {
    "net": ENet3D,
    "args": {"kernels": 16, "factor": 4},
}
models_params["AttentionENet3D"] = {
    "net": AttentionENet3D,
    "args": {"kernels": 16, "factor": 4},
}

optimizer_params: dict[str, dict[str, Any]] = {}
optimizer_params["adam"] = {
    "optim": torch.optim.Adam,
    "args": {"betas": (0.9, 0.999)},
}
optimizer_params["sgd"] = {"optim": torch.optim.SGD, "args": {}}


def precision_coef(p: Tensor, t: Tensor, eps: float = 1e-5) -> Tensor:
    """Computes Precision per class: TP / (TP + FP)."""
    sum_dims = tuple(range(2, p.dim()))
    tp = (p * t).sum(dim=sum_dims)
    pred_sum = p.sum(dim=sum_dims)
    return (tp + eps) / (pred_sum + eps)


def recall_coef(p: Tensor, t: Tensor, eps: float = 1e-5) -> Tensor:
    """Computes Recall per class: TP / (TP + FN)."""
    sum_dims = tuple(range(2, p.dim()))
    tp = (p * t).sum(dim=sum_dims)
    gt_sum = t.sum(dim=sum_dims)
    return (tp + eps) / (gt_sum + eps)


def img_transform(img):
    img = img[np.newaxis, ...]
    img = torch.tensor(img, dtype=torch.float32)
    return img


def gt_transform(K, img):
    img = torch.tensor(img, dtype=torch.int64)[None, ...]
    img = class2one_hot(img, K=K)
    return img[0]


def sliding_window_inference(
    inputs: Tensor,
    patch_size: tuple[int, int, int] | None,
    overlap: float,
    net: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    memory_format: torch.memory_format | None = None,
    loss_fn: nn.Module | None = None,
    gt: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, list[Tensor]]:
    B, C, D, H, W = inputs.shape

    if patch_size is None:
        patch_size = (D, H, W)

    pD, pH, pW = patch_size

    pad_d = max(0, pD - D)
    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        inputs = F.pad(inputs, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0)
        if gt is not None:
            gt = F.pad(gt, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0)
        _, _, p_D, p_H, p_W = inputs.shape
    else:
        p_D, p_H, p_W = D, H, W

    stride_d = max(1, int(pD * (1.0 - overlap)))
    stride_h = max(1, int(pH * (1.0 - overlap)))
    stride_w = max(1, int(pW * (1.0 - overlap)))

    def get_offsets(dim_len, patch_len, stride):
        if dim_len <= patch_len:
            return [0]
        offsets = list(range(0, dim_len - patch_len + 1, stride))
        if offsets[-1] + patch_len < dim_len:
            offsets.append(dim_len - patch_len)
        return offsets

    z_offsets = get_offsets(p_D, pD, stride_d)
    y_offsets = get_offsets(p_H, pH, stride_h)
    x_offsets = get_offsets(p_W, pW, stride_w)

    output_probs = None
    count_map = torch.zeros((B, 1, p_D, p_H, p_W), device="cpu", dtype=torch.float32)

    patch_losses = []
    patch_info_list = []

    for z in z_offsets:
        for y in y_offsets:
            for x in x_offsets:
                patch = inputs[:, :, z : z + pD, y : y + pH, x : x + pW]
                if memory_format is not None:
                    patch = patch.to(memory_format=memory_format)

                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    out = net(patch)
                    if isinstance(out, tuple):
                        out = out[0]
                    probs = F.softmax(out, dim=1)

                probs_cpu = probs.detach().float().cpu()

                if output_probs is None:
                    output_probs = torch.zeros(
                        (B, probs.shape[1], p_D, p_H, p_W),
                        device="cpu",
                        dtype=torch.float32,
                    )

                output_probs[:, :, z : z + pD, y : y + pH, x : x + pW] += probs_cpu
                count_map[:, :, z : z + pD, y : y + pH, x : x + pW] += 1.0

                if loss_fn is not None and gt is not None:
                    patch_gt = gt[:, :, z : z + pD, y : y + pH, x : x + pW]
                    if memory_format is not None:
                        patch_gt = patch_gt.to(memory_format=memory_format)

                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                    ):
                        p_loss_main, *p_loss_info = loss_fn(probs, patch_gt)

                    patch_losses.append(p_loss_main.detach().cpu())
                    if p_loss_info:
                        patch_info_list.append(
                            [info.detach().cpu() for info in p_loss_info]
                        )

    output_probs /= count_map
    final_output_probs = output_probs[:, :, :D, :H, :W]

    avg_loss = torch.stack(patch_losses).mean() if patch_losses else None

    avg_info = []
    if patch_info_list:
        num_metrics = len(patch_info_list[0])
        for m_idx in range(num_metrics):
            m_vals = torch.stack([p_info[m_idx] for p_info in patch_info_list])
            avg_info.append(m_vals.mean())

    return final_output_probs, avg_loss, avg_info


def evaluate_val_patches(
    net: nn.Module,
    img: Tensor,
    gt: Tensor,
    patch_size: tuple[int, int, int],
    overlap: float,
    loss_fn: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    memory_format: torch.memory_format | None = None,
    val_batch_size: int = 8,
) -> tuple[float, Tensor, Tensor, Tensor, list[Tensor]]:
    B, C, D, H, W = img.shape
    pD, pH, pW = patch_size

    pad_d = max(0, pD - D)
    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0)
        gt = F.pad(gt, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0)
        _, _, p_D, p_H, p_W = img.shape
    else:
        p_D, p_H, p_W = D, H, W

    stride_d = max(1, int(pD * (1.0 - overlap)))
    stride_h = max(1, int(pH * (1.0 - overlap)))
    stride_w = max(1, int(pW * (1.0 - overlap)))

    def get_offsets(dim_len, patch_len, stride):
        if dim_len <= patch_len:
            return [0]
        offsets = list(range(0, dim_len - patch_len + 1, stride))
        if offsets[-1] + patch_len < dim_len:
            offsets.append(dim_len - patch_len)
        return offsets

    z_offsets = get_offsets(p_D, pD, stride_d)
    y_offsets = get_offsets(p_H, pH, stride_h)
    x_offsets = get_offsets(p_W, pW, stride_w)

    patch_coords = [(z, y, x) for z in z_offsets for y in y_offsets for x in x_offsets]

    patch_losses = []
    patch_dices = []
    patch_precs = []
    patch_recs = []
    patch_info_list = []

    for idx in range(0, len(patch_coords), val_batch_size):
        batch_coords = patch_coords[idx : idx + val_batch_size]

        imgs_list = [
            img[:, :, z : z + pD, y : y + pH, x : x + pW] for (z, y, x) in batch_coords
        ]
        gts_list = [
            gt[:, :, z : z + pD, y : y + pH, x : x + pW] for (z, y, x) in batch_coords
        ]

        batch_img = torch.cat(imgs_list, dim=0)
        batch_gt = torch.cat(gts_list, dim=0)

        if memory_format is not None:
            batch_img = batch_img.to(memory_format=memory_format)
            batch_gt = batch_gt.to(memory_format=memory_format)

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            out = net(batch_img)
            if isinstance(out, tuple):
                out = out[0]
            probs = F.softmax(out, dim=1)
            p_loss, *p_info = loss_fn(probs, batch_gt)

        patch_seg = probs2one_hot(probs)
        p_dice = dice_coef(patch_seg, batch_gt)
        p_prec = precision_coef(patch_seg, batch_gt)
        p_rec = recall_coef(patch_seg, batch_gt)

        patch_losses.append(p_loss.detach().item())
        patch_dices.append(p_dice.detach().cpu())
        patch_precs.append(p_prec.detach().cpu())
        patch_recs.append(p_rec.detach().cpu())

        if p_info:
            patch_info_list.append([info.detach() for info in p_info])

    avg_loss = float(np.mean(patch_losses))
    avg_dice = torch.cat(patch_dices, dim=0).mean(dim=0, keepdim=True)
    avg_prec = torch.cat(patch_precs, dim=0).mean(dim=0, keepdim=True)
    avg_rec = torch.cat(patch_recs, dim=0).mean(dim=0, keepdim=True)

    avg_info = []
    if patch_info_list:
        num_metrics = len(patch_info_list[0])
        for m_idx in range(num_metrics):
            m_vals = torch.stack([p_info[m_idx] for p_info in patch_info_list])
            avg_info.append(m_vals.mean())

    return avg_loss, avg_dice, avg_prec, avg_rec, avg_info


def export_best_predictions(
    net: nn.Module,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    memory_format: torch.memory_format | None,
):
    print("\n>> Training complete. Exporting full 3D predictions for best model...")
    best_weights_path = args.dest / "bestweights.pt"
    if not best_weights_path.exists():
        print(">> No best weights file found. Skipping export.")
        return

    best_folder = args.dest / "best_epoch" / "val"
    best_folder.mkdir(parents=True, exist_ok=True)

    model_to_load = getattr(net, "_orig_mod", net)
    model_to_load.load_state_dict(
        torch.load(best_weights_path, map_location=device, weights_only=True)
    )
    net.eval()

    patch_size = tuple(args.patch_size) if args.patch_size else None

    with torch.no_grad():
        for i, data in tqdm_(
            enumerate(val_loader),
            total=len(val_loader),
            desc=">> Exporting Best 3D Volumes",
        ):
            img = data["images"].to(device, non_blocking=True)
            B = img.shape[0]

            pred_probs, _, _ = sliding_window_inference(
                inputs=img,
                patch_size=patch_size,
                overlap=args.overlap,
                net=net,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
                memory_format=memory_format,
            )

            predicted_class = probs2class(pred_probs)

            for b in range(B):
                vol = predicted_class[b].cpu().numpy().astype(np.uint8)
                vol = np.transpose(vol, (1, 2, 0))

                patient_affine = data["affine"][b].cpu().numpy()
                patient_orig_shape = data["orig_shape"][b].cpu().numpy()
                patient_stem = data["stems"][b]

                if args.resample:
                    raw_folder = "test" if val_loader.dataset.test_mode else "train"
                    patient_dir = (
                        Path("data") / args.dataset / raw_folder / patient_stem
                    )
                    orig_img_path = patient_dir / f"{patient_stem}.nii.gz"

                    orig_sitk = sitk.ReadImage(str(orig_img_path))
                    pred_sitk = sitk.GetImageFromArray(np.transpose(vol, (2, 0, 1)))
                    pred_sitk.SetSpacing(tuple(args.target_spacing))

                    resample_filter = sitk.ResampleImageFilter()
                    resample_filter.SetReferenceImage(orig_sitk)
                    resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
                    resample_filter.SetDefaultPixelValue(0)

                    resampled_sitk = resample_filter.Execute(pred_sitk)
                    resized_vol = sitk.GetArrayFromImage(resampled_sitk)
                    resized_vol = np.transpose(resized_vol, (1, 2, 0))

                    orig_nii = nib.load(str(orig_img_path))
                    nifti_img = nib.Nifti1Image(
                        resized_vol.astype(np.uint8), affine=orig_nii.affine
                    )
                else:
                    resized_vol = resize(
                        vol.astype(float),
                        tuple(patient_orig_shape),
                        order=0,
                        mode="constant",
                        preserve_range=True,
                        anti_aliasing=False,
                    ).astype(np.uint8)

                    nifti_img = nib.Nifti1Image(resized_vol, affine=patient_affine)

                nib.save(nifti_img, best_folder / f"{patient_stem}.nii.gz")

            del img, pred_probs, predicted_class
            torch.cuda.empty_cache()
            gc.collect()

    print(f">> Successfully exported 3D NIfTI volumes to: {best_folder}")


def build_scheduler(
    optimizer, warmup_epochs, total_epochs, first_cycle_epochs=7, eta_min=1e-6
):
    if optimizer is None:
        return None

    post_warmup_epochs = max(0, total_epochs - warmup_epochs)
    cycle1_epochs = min(first_cycle_epochs, post_warmup_epochs)
    cycle2_epochs = max(0, post_warmup_epochs - cycle1_epochs)

    schedulers = []
    milestones = []

    if warmup_epochs > 0 and total_epochs > warmup_epochs:
        schedulers.append(
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        )
        milestones.append(warmup_epochs)

    if cycle1_epochs > 0:
        schedulers.append(
            CosineAnnealingLR(optimizer, T_max=cycle1_epochs, eta_min=eta_min)
        )

    if cycle2_epochs > 0:
        milestones.append(
            milestones[-1] + cycle1_epochs if milestones else cycle1_epochs
        )
        schedulers.append(
            CosineAnnealingLR(optimizer, T_max=cycle2_epochs, eta_min=eta_min)
        )

    if not schedulers:
        return None

    if len(schedulers) == 1:
        return schedulers[0]

    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def setup(
    args,
) -> tuple[
    nn.Module,
    tuple[Any, Any],
    tuple[Any, Any],
    nn.Module,
    torch.device,
    DataLoader,
    DataLoader,
    int,
    bool,
    torch.memory_format | None,
]:
    if args.dest.exists():
        print(f">> Removing existing output directory: {args.dest}")
        rmtree(args.dest)
    args.dest.mkdir(parents=True, exist_ok=True)

    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    if gpu:
        if args.tf32:
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.allow_tf32 = True
            print(">> Enabled TF32 precision")

        torch.backends.cudnn.benchmark = True

    seed_everything(seed=args.seed, deterministic=args.deterministic)

    K: int = datasets_params[args.dataset]["K"]
    is_3d_model = "3D" in args.model

    if is_3d_model:
        net = models_params[args.model]["net"](
            in_dim=1, out_dim=K, **models_params[args.model]["args"]
        )
        z_window = 1
    else:
        z_window = models_params[args.model]["args"].get("z_window", 1)
        net = models_params[args.model]["net"](
            z_window, K, **models_params[args.model]["args"]
        )

    net.init_weights()
    net = net.to(device=device)

    memory_format = None
    if getattr(args, "channels_last", False):
        memory_format = torch.channels_last_3d if is_3d_model else torch.channels_last
        net = net.to(memory_format=memory_format)
        fmt_name = "channels_last_3d" if is_3d_model else "channels_last"
        print(f">> Converted model parameters to {fmt_name} memory format.")

    B: int = (
        args.batch_size
        if getattr(args, "batch_size", None) is not None
        else datasets_params[args.dataset]["B"]
    )

    if is_3d_model:
        if args.patch_size:
            summary_input_size = (B, 1, *tuple(args.patch_size))
        else:
            summary_input_size = (B, 1, 128, 256, 256)
    else:
        summary_input_size = (B, z_window, 512, 512)

    print("=== MODEL SUMMARY ===")
    try:
        net.eval()
        summary(net, input_size=summary_input_size, device=device.type)
    except Exception as e:
        print(f">> Model summary failed with input shape {summary_input_size}: {e}")

    if getattr(args, "compile", False):
        if hasattr(torch, "compile"):
            print(">> Compiling model with torch.compile()...")
            net = torch.compile(net)
        else:
            print(">> Warning: torch.compile is not supported on this PyTorch version.")

    if args.mode == "full":
        supervised_ids = list(range(K))
    elif args.mode in ["partial"] and args.dataset == "SEGTHOR":
        supervised_ids = [0, 1, 3, 4]
    else:
        raise ValueError(args.mode, args.dataset)

    loss_kwargs = {
        "use_focal": getattr(args, "use_focal", False),
        "alpha": getattr(args, "alpha", 0.6),
        "beta": getattr(args, "beta", 0.4),
        "idk": supervised_ids,
        "device": device,
        "legacy": getattr(args, "legacy_loss", False),
        "ema_decay": getattr(args, "ema_decay", 0.8),
        "dynamic_power": getattr(args, "dynamic_power", 2.0),
        "warmup_epochs": getattr(args, "dynamic_warmup", 3),
        "rampup_epochs": getattr(args, "rampup_epochs", 5),
        "schedule_type": getattr(args, "rampup_schedule", "linear"),
    }

    optim_kwargs = optimizer_params[args.optim]["args"].copy()
    if args.optim == "adam":
        optim_kwargs["fused"] = gpu and getattr(args, "fused", False)

    optimizer_net = optimizer_params[args.optim]["optim"](
        net.parameters(), lr=args.lr, **optim_kwargs
    )

    optimizer_loss = None
    if args.loss == "ce":
        if loss_kwargs["use_focal"]:
            loss_fn = FocalLoss(**loss_kwargs).to(device)
        else:
            loss_fn = CrossEntropy(**loss_kwargs).to(device)
    elif args.loss == "gdl":
        loss_fn = GeneralizedDice(**loss_kwargs).to(device)
    elif args.loss == "tversky":
        loss_fn = TverskyLoss(**loss_kwargs).to(device)
    elif args.loss == "compound":
        loss_fn = CompoundLoss(**loss_kwargs).to(device)
        loss_params = list(loss_fn.parameters())
        if len(loss_params) > 0:
            optimizer_loss = optimizer_params[args.optim]["optim"](
                loss_params,
                lr=1e-3,
                weight_decay=0.0,
                **optim_kwargs,
            )

    loss_fn.aux_loss_fn = (
        CrossEntropy(**loss_kwargs).to(device)
        if isinstance(loss_fn, CompoundLoss)
        else loss_fn
    )

    warmup_epochs = getattr(args, "warmup_epochs", 3)
    scheduler_net = build_scheduler(optimizer_net, warmup_epochs, args.epochs)

    root_dir = Path("data") / args.dataset
    DatasetClass = Segthor3DDataset if is_3d_model else SliceDataset

    dataset_kwargs = {
        "img_transform": img_transform,
        "gt_transform": partial(gt_transform, K),
        "debug": args.debug,
        "z_window": z_window,
        "resample": args.resample,
    }

    if is_3d_model:
        dataset_kwargs["target_spacing"] = tuple(args.target_spacing)
        dataset_kwargs["patch_size"] = (
            tuple(args.patch_size) if args.patch_size else None
        )
        dataset_kwargs["samples_per_volume"] = args.samples_per_volume

    num_workers = 2
    if os.environ.get("IS_SNELLIUS", 0) == 1:
        num_workers = 4
    elif is_3d_model:
        num_workers = 1

    train_set = DatasetClass(
        "train",
        root_dir,
        augment=args.augment,
        drop_empty=args.drop_empty,
        **dataset_kwargs,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=B,
        num_workers=2 * num_workers,
        prefetch_factor=min(2, num_workers),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
        shuffle=True,
        pin_memory=gpu,
        drop_last=True if len(train_set) > B else False,
    )

    val_set = DatasetClass(
        "val",
        root_dir,
        **dataset_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1 if is_3d_model else B,
        num_workers=2 * num_workers,
        prefetch_factor=min(2, num_workers),
        shuffle=False,
        pin_memory=gpu,
    )

    return (
        net,
        (optimizer_net, optimizer_loss),
        (scheduler_net, None),
        loss_fn,
        device,
        train_loader,
        val_loader,
        K,
        is_3d_model,
        memory_format,
    )


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    (
        net,
        (optimizer_net, optimizer_loss),
        (scheduler_net, scheduler_loss),
        loss_fn,
        device,
        train_loader,
        val_loader,
        K,
        is_3d_model,
        memory_format,
    ) = setup(args)

    amp_enabled: bool = args.amp != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    scaler = torch.amp.GradScaler(
        device.type, enabled=(amp_enabled and args.amp == "fp16")
    )

    print(f">> Mixed Precision (AMP): {args.amp.upper()} (Enabled: {amp_enabled})")

    patch_size = tuple(args.patch_size) if args.patch_size else None

    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
    log_prec_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
    log_rec_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0.0

    for e in range(args.epochs):
        if hasattr(loss_fn, "update_scheduled_weights"):
            active_w = loss_fn.update_scheduled_weights(epoch=e)
            if active_w is not None and getattr(args, "rampup_epochs", 0) > 0:
                print(
                    f">> Epoch {e:02d} Active Class Weights (Rampup): {active_w.cpu().numpy().round(3).tolist()}"
                )

        for m in ["train", "val"]:
            match m:
                case "train":
                    net.train()
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                    is_train = True
                case "val":
                    net.eval()
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val
                    is_train = False

            with cm():
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data["images"].to(device, non_blocking=True)
                    gt = data["gts"].to(device, non_blocking=True)

                    if is_train:
                        if memory_format is not None:
                            img = img.to(memory_format=memory_format)
                            gt = gt.to(memory_format=memory_format)

                        optimizer_net.zero_grad(set_to_none=True)
                        if optimizer_loss:
                            optimizer_loss.zero_grad(set_to_none=True)

                        with torch.autocast(
                            device_type=device.type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
                            out = net(img)
                            if isinstance(out, tuple):
                                pred_logits, aux_logits = out
                                aux_probs = F.softmax(aux_logits, dim=1)
                                loss_aux, *_ = loss_fn.aux_loss_fn(aux_probs, gt)
                            else:
                                pred_logits = out
                                loss_aux = 0.0

                            pred_probs = F.softmax(1 * pred_logits, dim=1)
                            loss_main, *loss_info = loss_fn(pred_probs, gt)

                            if is_3d_model:
                                loss = (
                                    loss_main + 0.1 * loss_aux
                                    if isinstance(out, tuple)
                                    else loss_main
                                )
                            else:
                                loss = (
                                    loss_main + 0.4 * loss_aux
                                    if isinstance(out, tuple)
                                    else loss_main
                                )

                        B = img.shape[0]

                        with torch.no_grad():
                            pred_seg = probs2one_hot(pred_probs)
                            log_dice[e, j : j + B, :] = dice_coef(pred_seg, gt)

                        log_loss[e, i] = loss.item()

                        scaler.scale(loss).backward()

                        if args.clip_grad > 0:
                            scaler.unscale_(optimizer_net)
                            torch.nn.utils.clip_grad_norm_(
                                net.parameters(), max_norm=args.clip_grad
                            )
                            if optimizer_loss:
                                scaler.unscale_(optimizer_loss)
                                torch.nn.utils.clip_grad_norm_(
                                    loss_fn.parameters(), max_norm=args.clip_grad
                                )

                        scaler.step(optimizer_net)
                        if optimizer_loss:
                            scaler.step(optimizer_loss)

                        scaler.update()

                    else:
                        # VALIDATION
                        B = img.shape[0]
                        if is_3d_model and patch_size:
                            v_loss, v_dice, v_prec, v_rec, loss_info = (
                                evaluate_val_patches(
                                    net=net,
                                    img=img,
                                    gt=gt,
                                    patch_size=patch_size,
                                    overlap=args.val_overlap,
                                    loss_fn=loss_fn,
                                    device=device,
                                    amp_dtype=amp_dtype,
                                    amp_enabled=amp_enabled,
                                    memory_format=memory_format,
                                    val_batch_size=args.val_batch_size,
                                )
                            )
                            log_loss[e, i] = v_loss
                            log_dice[e, j : j + B, :] = v_dice
                            log_prec_val[e, j : j + B, :] = v_prec
                            log_rec_val[e, j : j + B, :] = v_rec
                        else:
                            if memory_format is not None:
                                img = img.to(memory_format=memory_format)
                            with torch.autocast(
                                device_type=device.type,
                                dtype=amp_dtype,
                                enabled=amp_enabled,
                            ):
                                out = net(img)
                                if isinstance(out, tuple):
                                    out = out[0]
                                pred_probs = F.softmax(1 * out, dim=1)

                            loss_main, *loss_info = loss_fn(pred_probs, gt)
                            log_loss[e, i] = loss_main.item()

                            with torch.no_grad():
                                pred_seg = probs2one_hot(pred_probs)
                                log_dice[e, j : j + B, :] = dice_coef(pred_seg, gt)
                                log_prec_val[e, j : j + B, :] = precision_coef(
                                    pred_seg, gt
                                )
                                log_rec_val[e, j : j + B, :] = recall_coef(pred_seg, gt)

                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=UserWarning)
                                predicted_class: Tensor = probs2class(pred_probs)
                                mult: float = 63.0 if K == 5 else (255.0 / (K - 1))

                                save_dir = args.dest / f"iter{e:03d}" / m
                                save_dir.mkdir(parents=True, exist_ok=True)

                                if predicted_class.dim() == 4:
                                    mid_z = predicted_class.shape[1] // 2
                                    save_images(
                                        predicted_class[:, mid_z] * mult,
                                        data["stems"],
                                        save_dir,
                                    )
                                else:
                                    save_images(
                                        predicted_class * mult,
                                        data["stems"],
                                        save_dir,
                                    )

                        del img, gt
                        torch.cuda.empty_cache()

                    j += B
                    postfix_dict: dict[str, str] = {
                        "Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                        "Loss": f"{log_loss[e, :i + 1].mean():5.2e}",
                    }
                    if isinstance(loss_fn, CompoundLoss) and loss_info:
                        sigma_ce = torch.exp(0.5 * loss_fn.s_ce).item()
                        sigma_gdl = torch.exp(0.5 * loss_fn.s_gdl).item()
                        postfix_dict |= {
                            "CE": f"{loss_info[0].item():5.2e}",
                            "GDL": f"{loss_info[1].item():5.2e}",
                            "s_ce": f"{sigma_ce:.2f}",
                            "s_gdl": f"{sigma_gdl:.2f}",
                        }
                    if K > 2:
                        postfix_dict |= {
                            f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                            for k in range(1, K)
                        }
                    tq_iter.set_postfix(postfix_dict)

        # Dynamic weight update based on Precision and Recall feedback
        val_dice_per_class = log_dice_val[e].mean(dim=0)
        val_prec_per_class = log_prec_val[e].mean(dim=0)
        val_rec_per_class = log_rec_val[e].mean(dim=0)

        if getattr(args, "dynamic_weights", False) and hasattr(
            loss_fn, "update_dynamic_class_weights"
        ):
            updated_w = loss_fn.update_dynamic_class_weights(
                val_dice_per_class,
                val_prec_per_class,
                val_rec_per_class,
                epoch=e,
            )
            if updated_w is not None and e >= getattr(args, "dynamic_warmup", 3):
                print(
                    f">> Updated Dynamic Class Weights for Epoch {e+1}: {updated_w.cpu().numpy().round(3).tolist()}"
                )

        scheduler_net.step()
        if scheduler_loss:
            scheduler_loss.step()

        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        print(
            f">> Epoch: {e} | LR: {scheduler_net.get_last_lr()[0]:.2e} | DSC: {current_dice:05.3f}"
        )

        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", "w") as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)

            if not is_3d_model:
                iter_dir = args.dest / f"iter{e:03d}"
                if iter_dir.exists():
                    copytree(iter_dir, Path(best_folder))

            model_to_save = getattr(net, "_orig_mod", net)
            torch.save(model_to_save, args.dest / "bestmodel.pkl")
            torch.save(model_to_save.state_dict(), args.dest / "bestweights.pt")

    if is_3d_model:
        export_best_predictions(
            net=net,
            val_loader=val_loader,
            args=args,
            device=device,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
            memory_format=memory_format,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--dataset", default="SEGTHOR", choices=datasets_params.keys())
    parser.add_argument(
        "--augment",
        action="store_true",
    )
    parser.add_argument(
        "--drop_empty",
        action="store_true",
    )
    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--patch_size",
        nargs=3,
        default=None,
        type=int,
    )
    parser.add_argument(
        "--samples_per_volume",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--overlap",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--val_overlap",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--val_batch_size",
        default=8,
        type=int,
    )
    parser.add_argument("--model", default="ENet3D", choices=models_params.keys())
    parser.add_argument("--optim", default="adam", choices=optimizer_params.keys())
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument(
        "--warmup-epochs",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--loss",
        default="ce",
        choices=["ce", "gdl", "tversky", "compound"],
    )
    parser.add_argument(
        "--use_focal",
        action="store_true",
    )
    parser.add_argument(
        "--alpha",
        default=0.6,
        type=float,
        help="Tversky Loss False Positive penalty multiplier (higher boosts precision).",
    )
    parser.add_argument(
        "--beta",
        default=0.4,
        type=float,
        help="Tversky Loss False Negative penalty multiplier.",
    )
    parser.add_argument(
        "--legacy_loss",
        action="store_true",
    )
    parser.add_argument(
        "--rampup_epochs",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--rampup_schedule",
        default="linear",
        choices=["linear", "cosine"],
    )
    parser.add_argument(
        "--dynamic_weights",
        action="store_true",
    )
    parser.add_argument(
        "--ema_decay",
        default=0.8,
        type=float,
    )
    parser.add_argument(
        "--dynamic_power",
        default=2.0,
        type=float,
    )
    parser.add_argument(
        "--dynamic_warmup",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--clip-grad",
        default=1.0,
        type=float,
    )
    parser.add_argument("--mode", default="full", choices=["partial", "full"])
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--fused",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--amp",
        default="bf16",
        choices=["none", "fp16", "bf16"],
    )
    parser.add_argument(
        "--channels_last",
        action="store_true",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
    )
    parser.add_argument(
        "--no-tf32",
        dest="tf32",
        default=True,
        action="store_false",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
    )
    parser.add_argument(
        "--target_spacing",
        nargs=3,
        default=[1.0, 1.0, 1.0],
        type=float,
    )

    args = parser.parse_args()

    pprint(args)

    runTraining(args)


if __name__ == "__main__":
    main()
