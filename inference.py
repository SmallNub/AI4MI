#!/usr/bin/env python3

import argparse
import warnings
from pathlib import Path
from pprint import pprint
from functools import partial
from typing import Any

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (
    class2one_hot,
    probs2class,
    tqdm_,
    save_images,
)

datasets_params: dict[str, dict[str, Any]] = {
    "TOY2": {"K": 2, "net": shallowCNN, "B": 2, "kernels": 8, "factor": 2},
    "SEGTHOR": {"K": 5, "net": ENet, "B": 8, "kernels": 8, "factor": 2},
    "SEGTHOR_CLEAN": {"K": 5, "net": ENet, "B": 8, "kernels": 8, "factor": 2},
}


def img_transform(img):
    img = img.convert("L")
    img = np.array(img)[np.newaxis, ...]
    img = img / 255  # max <= 1
    return torch.tensor(img, dtype=torch.float32)


def gt_transform(K, img):
    img = np.array(img)[...]
    img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
    img = torch.tensor(img, dtype=torch.int64)[None, ...]
    img = class2one_hot(img, K=K)
    return img[0]


def run_inference(args):
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} for inference")

    config = datasets_params[args.dataset]
    K: int = config["K"]
    kernels: int = config.get("kernels", 8)
    factor: int = config.get("factor", 2)
    B: int = config["B"]

    # Instantiate network architecture
    net = config["net"](1, K, kernels=kernels, factor=factor)

    # Load weights or full model
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    print(f">> Loading model/weights from {weights_path}")
    if weights_path.suffix in [".pt", ".pth"]:
        state_dict = torch.load(weights_path, map_location=device)
        net.load_state_dict(state_dict)
    else:
        net = torch.load(weights_path, map_location=device)

    net.to(device)
    net.eval()

    # Prepare dataset loader
    root_dir = Path("data") / args.dataset
    dataset = SliceDataset(
        args.split,
        root_dir,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        debug=args.debug,
    )
    loader = DataLoader(
        dataset,
        batch_size=B,
        num_workers=5,
        shuffle=False,
    )

    args.dest.mkdir(parents=True, exist_ok=True)
    mult: int = 63 if K == 5 else (255 / (K - 1))

    print(f">> Running inference on '{args.split}' split ({len(dataset)} items)...")
    with torch.no_grad():
        for data in tqdm_(loader, desc=f">> Inference ({args.split})"):
            img = data["images"].to(device)

            # Adapt input dimensions for 3D networks if input is 2D slice (B, C, H, W) -> (B, C, 1, H, W)
            is_input_2d = (img.ndim == 4)
            if args.is_3d and is_input_2d:
                img = img.unsqueeze(2)  # Add Depth dimension: (B, C, 1, H, W)

            pred_logits = net(img)
            pred_probs = F.softmax(pred_logits, dim=1)  # softmax along class dimension

            # Class index map across channels (works for both 4D and 5D tensors)
            predicted_class = probs2class(pred_probs)

            # Squeeze extra depth dimension if network returned 3D with D=1
            if predicted_class.ndim == 4 and is_input_2d:
                # Shape is (B, 1, H, W) -> squeeze out depth dimension 1
                predicted_class = predicted_class.squeeze(1)

            # Save 2D outputs
            if predicted_class.ndim == 3:  # Batch of 2D slices: (B, H, W)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    save_images(
                        predicted_class * mult,
                        data["stems"],
                        args.dest,
                    )
            elif predicted_class.ndim == 4:  # Batch of 3D volumes: (B, D, H, W)
                # Unroll 3D volume into individual slice PNGs
                for b, stem in enumerate(data["stems"]):
                    vol_pred = predicted_class[b]  # (D, H, W)
                    for d_idx in range(vol_pred.shape[0]):
                        slice_stem = f"{stem}_{d_idx:04d}"
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            save_images(
                                vol_pred[d_idx : d_idx + 1] * mult,
                                [slice_stem],
                                args.dest,
                            )

    print(f">> Successfully saved predictions to {args.dest}")


def main():
    parser = argparse.ArgumentParser(description="Inference script supporting 2D and 3D models")

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to saved model weights or pickled model file",
    )
    parser.add_argument(
        "--dataset",
        default="SEGTHOR",
        choices=datasets_params.keys(),
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to run inference on",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination directory for saved output PNG masks",
    )
    parser.add_argument(
        "--is_3d",
        action="store_true",
        help="Flag to enable 3D input unsqueezing and volume unrolling for 3D architectures",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only 10 samples for quick logic verification",
    )

    args = parser.parse_args()
    pprint(args)

    run_inference(args)


if __name__ == "__main__":
    main()