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

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from shutil import copytree, rmtree

import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)

from functools import partial
from skimage.transform import resize

from dataset import SliceDataset, Segthor3DDataset
from ShallowNet import shallowCNN
from ENet import ENet, AttentionENet, SpatialENet, CBAMENet, LateFusionENet
from ImprovedENet import ImprovedENet
from ENet3D import ENet3D, AttentionENet3D
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

from losses import CrossEntropy, FocalLoss, GeneralizedDice, CompoundLoss

datasets_params: dict[str, dict[str, Any]] = {}
datasets_params["TOY2"] = {"K": 2, "B": 2}
datasets_params["SEGTHOR"] = {"K": 5, "B": 8}
datasets_params["SEGTHOR_CLEAN"] = {"K": 5, "B": 8}
datasets_params["segthor_train_full"] = {"K": 5, "B": 4}

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
    "args": {"kernels": 8, "factor": 2, "z_window": 5},
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
optimizer_params["adam"] = {"optim": torch.optim.Adam, "args": {"betas": (0.9, 0.999)}}
optimizer_params["sgd"] = {"optim": torch.optim.SGD, "args": {}}


def img_transform(img):
    img = img.convert("L")
    img = np.array(img)[np.newaxis, ...]
    img = img / 255  # max <= 1
    img = torch.tensor(img, dtype=torch.float32)
    return img


def gt_transform(K, img):
    img = np.array(img)[...]
    img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
    img = torch.tensor(img, dtype=torch.int64)[
        None, ...
    ]  # Add one dimension to simulate batch
    img = class2one_hot(img, K=K)
    return img[0]


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    if optimizer is None:
        return None
    if warmup_epochs > 0 and total_epochs > warmup_epochs:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=8,
            T_mult=2,
            eta_min=1e-6,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=8,
            T_mult=2,
            eta_min=1e-6,
        )


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
]:
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
        net = models_params[args.model]["net"](in_dim=1, out_dim=K, **models_params[args.model]["args"])
    else:
        z_window = models_params[args.model]["args"].get("z_window", 1)
        net = models_params[args.model]["net"](z_window, K, **models_params[args.model]["args"])

    net.init_weights()
    net.to(device)

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
        "use_focal": args.use_focal if hasattr(args, "use_focal") else False,
        "idk": supervised_ids,
        "device": device,
    }

    optimizer_net = optimizer_params[args.optim]["optim"](
        net.parameters(), lr=args.lr, **optimizer_params[args.optim]["args"]
    )

    optimizer_loss = None
    if args.loss == "ce":
        if loss_kwargs["use_focal"]:
            loss_fn = FocalLoss(**loss_kwargs)
        else:
            loss_fn = CrossEntropy(**loss_kwargs)
    elif args.loss == "gdl":
        loss_fn = GeneralizedDice(**loss_kwargs)
    elif args.loss == "compound":
        loss_fn = CompoundLoss(**loss_kwargs).to(device)
        loss_params = list(loss_fn.parameters())
        if len(loss_params) > 0:
            optimizer_loss = optimizer_params[args.optim]["optim"](
                loss_params,
                lr=1e-3,
                weight_decay=0.0,
                **optimizer_params[args.optim]["args"],
            )

    warmup_epochs = getattr(args, "warmup_epochs", 5)

    scheduler_net = build_scheduler(optimizer_net, warmup_epochs, args.epochs)

    B: int = args.batch_size if hasattr(args, "batch_size") else datasets_params[args.dataset]["B"]
    root_dir = Path("data") / args.dataset

    DatasetClass = Segthor3DDataset if is_3d_model else SliceDataset

    train_set = DatasetClass(
        "train",
        root_dir,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        augment=args.augment,
        drop_empty=args.drop_empty,
        debug=args.debug,
        z_window=z_window,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=B,
        num_workers=4 if is_3d_model else 8,
        prefetch_factor=2,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
        shuffle=True,
        pin_memory=gpu,
        drop_last=True if len(train_set) > B else False,
    )

    val_set = DatasetClass(
        "val",
        root_dir,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        debug=args.debug,
        z_window=z_window,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=B,
        num_workers=4 if is_3d_model else 8,
        prefetch_factor=2,
        shuffle=False,
        pin_memory=gpu,
    )

    args.dest.mkdir(parents=True, exist_ok=True)

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
    ) = setup(args)

    amp_enabled: bool = args.amp != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    scaler = torch.amp.GradScaler(
        device.type, enabled=(amp_enabled and args.amp == "fp16")
    )

    print(f">> Mixed Precision (AMP): {args.amp.upper()} (Enabled: {amp_enabled})")

    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    for e in range(args.epochs):
        val_3d_predictions = []  # Store 3D volumes in memory if is_3d_model is True

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
                        optimizer_net.zero_grad(set_to_none=True)
                        if optimizer_loss:
                            optimizer_loss.zero_grad(set_to_none=True)

                    assert 0 <= img.min() and img.max() <= 1
                    B = img.shape[0]

                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                    ):
                        pred_logits = net(img)
                        pred_probs = F.softmax(1 * pred_logits, dim=1)
                        loss, *loss_info = loss_fn(pred_probs, gt)

                    with torch.no_grad():
                        pred_seg = probs2one_hot(pred_probs)
                        log_dice[e, j : j + B, :] = dice_coef(pred_seg, gt)

                    log_loss[e, i] = loss.item()

                    if is_train:
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

                    if m == "val":
                        if is_3d_model:
                            predicted_class: Tensor = probs2class(pred_probs)  # Shape: [B, Z, H, W]
                            for b in range(B):
                                vol = predicted_class[b].cpu().numpy().astype(np.uint8)
                                vol = np.transpose(vol, (1, 2, 0))  # Convert [Z, H, W] to [H, W, Z]
                                
                                patient_affine = data["affine"][b].cpu().numpy()
                                patient_orig_shape = data["orig_shape"][b].cpu().numpy()  # [H, W, Z]
                                
                                val_3d_predictions.append((vol, data["stems"][b], patient_affine, patient_orig_shape))
                        else:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=UserWarning)
                                predicted_class: Tensor = probs2class(pred_probs)
                                mult: int = 63 if K == 5 else (255 / (K - 1))

                                if predicted_class.dim() == 4:
                                    mid_z = predicted_class.shape[1] // 2
                                    save_images(
                                        predicted_class[:, mid_z] * mult,
                                        data["stems"],
                                        args.dest / f"iter{e:03d}" / m,
                                    )
                                else:
                                    save_images(
                                        predicted_class * mult,
                                        data["stems"],
                                        args.dest / f"iter{e:03d}" / m,
                                    )

                    j += B
                    postfix_dict: dict[str, str] = {
                        "Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                        "Loss": f"{log_loss[e, :i + 1].mean():5.2e}",
                    }
                    if isinstance(loss_fn, CompoundLoss):
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

            if is_3d_model:
                best_folder.mkdir(parents=True, exist_ok=True)
                for vol_3d, stem, patient_affine, orig_shape in val_3d_predictions:
                    
                    # Rescale prediction back to the patient's native dimensions
                    resized_vol = resize(
                        vol_3d.astype(float),
                        tuple(orig_shape),
                        order=0,  # Nearest neighbor is mandatory for segmentation masks
                        mode="constant",
                        preserve_range=True,
                        anti_aliasing=False
                    ).astype(np.uint8)
                    
                    # Save using the true patient affine and native shape
                    nifti_img = nib.Nifti1Image(resized_vol, affine=patient_affine)
                    nib.save(nifti_img, best_folder / f"{stem}.nii.gz")
            else:
                copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            model_to_save = getattr(net, "_orig_mod", net)
            torch.save(model_to_save, args.dest / "bestmodel.pkl")
            torch.save(model_to_save.state_dict(), args.dest / "bestweights.pt")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--dataset", default="SEGTHOR", choices=datasets_params.keys())
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentations.",
    )
    parser.add_argument(
        "--drop_empty",
        action="store_true",
        help="Drop slices with no target labels (1, 2, 3, 4) during training.",
    )
    parser.add_argument(
        "--drop_empty",
        action="store_true",
        help="Drop slices with no target labels (1, 2, 3, 4) during training.",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size (set to 1 or 2 for 3D volumes to prevent CUDA OOM).",
    )
    parser.add_argument("--model", default="ENet3D", choices=models_params.keys())
    parser.add_argument("--optim", default="adam", choices=optimizer_params.keys())
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument(
        "--warmup-epochs",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--loss",
        default="ce",
        choices=["ce", "gdl", "compound"],
    )
    parser.add_argument(
        "--use_focal",
        action="store_true",
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
        "--amp",
        default="bf16",
        choices=["none", "fp16", "bf16"],
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

    args = parser.parse_args()

    pprint(args)

    runTraining(args)


if __name__ == "__main__":
    main()