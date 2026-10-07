#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

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


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import simplex, sset

EPS = 1e-4
# Adjusted default background weight from 0.2 to 0.5 to balance false positive penalty
DEFAULT_WEIGHTS = [0.5, 2.0, 1.0, 2.5, 1.2]


class CrossEntropy(nn.Module):
    def __init__(
        self,
        class_weights: list[float] | Tensor = None,
        legacy: bool = False,
        ema_decay: float = 0.8,
        dynamic_power: float = 2.0,
        warmup_epochs: int = 3,
        rampup_epochs: int = 5,
        schedule_type: str = "linear",
        **kwargs,
    ):
        super().__init__()
        self.idk = kwargs["idk"]
        self.eps = EPS
        self.legacy = legacy
        self.ema_decay = ema_decay
        self.dynamic_power = dynamic_power
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        self.schedule_type = schedule_type

        if class_weights is None:
            class_weights = DEFAULT_WEIGHTS

        device = kwargs.get("device", "cpu")
        if isinstance(class_weights, Tensor):
            target_tensor = class_weights.detach().to(
                dtype=torch.float32, device=device
            )
        else:
            target_tensor = torch.tensor(
                class_weights, dtype=torch.float32, device=device
            )

        self.register_buffer("target_weights", target_tensor)

        initial_weights = (
            torch.ones_like(target_tensor)
            if self.rampup_epochs > 0
            else target_tensor.clone()
        )
        self.register_buffer("weights", initial_weights)
        self.register_buffer("ema_prec", None, persistent=False)
        self.register_buffer("ema_rec", None, persistent=False)
        self.register_buffer("ema_val_dice", None, persistent=False)

        print(
            f"Initialized {self.__class__.__name__} (legacy={self.legacy}) "
            f"with target_weights={self.target_weights.tolist()}, rampup_epochs={self.rampup_epochs}"
        )

    def set_weights(self, new_weights: Tensor):
        """Update current active class weights."""
        self.weights.copy_(
            new_weights.detach().to(dtype=torch.float32, device=self.weights.device)
        )

    def update_scheduled_weights(self, epoch: int) -> Tensor:
        """Ramps up class weights from uniform (1:1) to target weights over rampup_epochs."""
        if self.rampup_epochs <= 0:
            self.set_weights(self.target_weights)
            return self.weights

        progress = min(1.0, max(0.0, float(epoch) / float(self.rampup_epochs)))

        if self.schedule_type == "cosine":
            alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
        else:  # linear
            alpha = progress

        uniform_weights = torch.ones_like(self.target_weights)
        current_weights = (1.0 - alpha) * uniform_weights + alpha * self.target_weights
        self.set_weights(current_weights)
        return current_weights

    def update_dynamic_class_weights(
        self,
        val_dice_all_classes: Tensor,
        val_prec_all_classes: Tensor | None = None,
        val_rec_all_classes: Tensor | None = None,
        epoch: int = 0,
    ) -> Tensor | None:
        """Dynamically updates target weights based on Precision and Recall feedback."""
        if self.legacy:
            return None

        if epoch < self.warmup_epochs:
            return self.update_scheduled_weights(epoch)

        device = self.weights.device

        if val_prec_all_classes is not None and val_rec_all_classes is not None:
            p = val_prec_all_classes.detach().to(device)
            r = val_rec_all_classes.detach().to(device)

            if self.ema_prec is None:
                self.ema_prec = p.clone()
                self.ema_rec = r.clone()
            else:
                self.ema_prec = (
                    self.ema_decay * self.ema_prec + (1.0 - self.ema_decay) * p
                )
                self.ema_rec = (
                    self.ema_decay * self.ema_rec + (1.0 - self.ema_decay) * r
                )

            new_target = self.target_weights.clone()
            overseg_penalty = 0.0

            # Foreground classes (indices 1 to K-1)
            for c in range(1, len(self.target_weights)):
                delta = self.ema_rec[c] - self.ema_prec[c]
                if delta > 0.05:
                    # Recall > Precision: Over-segmentation -> Reduce class weight
                    new_target[c] -= 0.05 * delta
                    overseg_penalty += delta
                elif delta < -0.05:
                    # Precision > Recall: Under-segmentation -> Increase class weight
                    new_target[c] += 0.05 * abs(delta)

                new_target[c] = torch.clamp(new_target[c], min=0.5, max=3.0)

            # Adjust Background weight (index 0)
            if overseg_penalty > 0:
                avg_overseg = overseg_penalty / (len(self.target_weights) - 1)
                new_target[0] += 0.05 * avg_overseg
            else:
                if new_target[0] > 0.5:
                    new_target[0] -= 0.01

            new_target[0] = torch.clamp(new_target[0], min=0.4, max=1.2)
            self.target_weights.copy_(new_target)
        else:
            val_dice = val_dice_all_classes.detach().to(device)
            if self.ema_val_dice is None:
                self.ema_val_dice = val_dice.clone()
            else:
                self.ema_val_dice = (
                    self.ema_decay * self.ema_val_dice
                    + (1.0 - self.ema_decay) * val_dice
                )

            raw_weights = torch.pow(1.0 - self.ema_val_dice, self.dynamic_power)
            normalized_weights = raw_weights / torch.clamp(
                torch.mean(raw_weights), min=self.eps
            )
            clamped_weights = torch.clamp(normalized_weights, min=0.5, max=2.5)
            self.target_weights.copy_(clamped_weights)

        return self.update_scheduled_weights(epoch)

    def forward(self, pred_softmax: Tensor, weak_target: Tensor) -> tuple[Tensor, list]:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        p = pred_softmax[:, self.idk, ...].float()
        t = weak_target[:, self.idk, ...].float()

        if self.legacy:
            w = self.weights[self.idk].view(1, len(self.idk), *([1] * (p.dim() - 2)))
            p_clamped = p.clamp(min=self.eps, max=1.0 - self.eps)
            log_p = p_clamped.log()

            weighted_loss = -(t * log_p * w).sum()
            normalizer = torch.clamp((t * w).sum(), min=self.eps)

            loss = weighted_loss / normalizer
            return loss, []
        else:
            w = self.weights[self.idk]
            p_clamped = p.clamp(min=self.eps, max=1.0 - self.eps)
            log_p = p_clamped.log()

            sum_dims = (0,) + tuple(range(2, p.dim()))

            per_class_numerator = -(t * log_p).sum(dim=sum_dims)
            per_class_denominator = torch.clamp(t.sum(dim=sum_dims), min=self.eps)
            per_class_loss = per_class_numerator / per_class_denominator

            weighted_loss = (per_class_loss * w).sum() / torch.clamp(
                w.sum(), min=self.eps
            )
            return weighted_loss, []


class FocalLoss(CrossEntropy):
    def __init__(
        self,
        gamma: float = 1.5,
        class_weights: list[float] | Tensor = None,
        legacy: bool = False,
        **kwargs,
    ):
        super().__init__(class_weights=class_weights, legacy=legacy, **kwargs)
        self.gamma = gamma

    def forward(self, pred_softmax: Tensor, weak_target: Tensor) -> tuple[Tensor, list]:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        p = pred_softmax[:, self.idk, ...].float()
        t = weak_target[:, self.idk, ...].float()

        p_clamped = p.clamp(min=self.eps, max=1.0 - self.eps)
        log_p = p_clamped.log()
        focal_weight = ((1.0 - p_clamped).clamp(min=self.eps)) ** self.gamma

        if self.legacy:
            w = self.weights[self.idk].view(1, len(self.idk), *([1] * (p.dim() - 2)))
            weighted_loss = -(t * focal_weight * log_p * w).sum()
            normalizer = torch.clamp((t * w).sum(), min=self.eps)
            loss = weighted_loss / normalizer
            return loss, []
        else:
            w = self.weights[self.idk]
            sum_dims = (0,) + tuple(range(2, p.dim()))

            per_class_numerator = -(t * focal_weight * log_p).sum(dim=sum_dims)
            per_class_denominator = torch.clamp(t.sum(dim=sum_dims), min=self.eps)
            per_class_loss = per_class_numerator / per_class_denominator

            weighted_loss = (per_class_loss * w).sum() / torch.clamp(
                w.sum(), min=self.eps
            )
            return weighted_loss, []


class GeneralizedDice(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.idk = kwargs["idk"]
        self.eps = EPS
        self.smooth = 1e-1

    def forward(self, pred_softmax: Tensor, weak_target: Tensor) -> tuple[Tensor, list]:
        p = pred_softmax.float()
        t = weak_target.float()

        sum_dims = (0,) + tuple(range(2, p.dim()))
        volumes = torch.sum(t, dim=sum_dims)

        v_frac = volumes / torch.clamp(torch.sum(volumes), min=self.eps)
        weights = 1.0 / (torch.square(v_frac) + self.smooth)

        intersection = torch.sum(p * t, dim=sum_dims)
        cardinality = torch.sum(p, dim=sum_dims) + volumes

        weights = weights[self.idk]
        intersection = intersection[self.idk]
        cardinality = cardinality[self.idk]

        gdl_num = torch.sum(weights * intersection)
        gdl_den = torch.clamp((weights * cardinality).sum(), min=self.eps)

        gdl = 1.0 - (2.0 * gdl_num / gdl_den)
        return gdl, []


class MacroDiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.idk = kwargs["idk"]
        self.eps = EPS

    def forward(self, pred_softmax: Tensor, weak_target: Tensor) -> tuple[Tensor, list]:
        p = pred_softmax[:, self.idk, ...].float()
        t = weak_target[:, self.idk, ...].float()

        sum_dims = (0,) + tuple(range(2, p.dim()))

        intersection = torch.sum(p * t, dim=sum_dims)
        cardinality = torch.sum(p, dim=sum_dims) + torch.sum(t, dim=sum_dims)

        dice_per_class = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        loss = 1.0 - dice_per_class.mean()
        return loss, []


class TverskyLoss(nn.Module):
    """
    Tversky Loss with tunable alpha (FP penalty) and beta (FN penalty).
    alpha = 0.6, beta = 0.4 penalizes False Positives to boost Precision.
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.4, **kwargs):
        super().__init__()
        self.idk = kwargs["idk"]
        self.alpha = alpha
        self.beta = beta
        self.eps = EPS

    def forward(self, pred_softmax: Tensor, weak_target: Tensor) -> tuple[Tensor, list]:
        p = pred_softmax[:, self.idk, ...].float()
        t = weak_target[:, self.idk, ...].float()

        sum_dims = (0,) + tuple(range(2, p.dim()))

        tp = torch.sum(p * t, dim=sum_dims)
        fp = torch.sum(p * (1.0 - t), dim=sum_dims)
        fn = torch.sum((1.0 - p) * t, dim=sum_dims)

        tversky_per_class = (tp + self.eps) / (
            tp + self.alpha * fp + self.beta * fn + self.eps
        )
        loss = 1.0 - tversky_per_class.mean()
        return loss, []


class CompoundLoss(nn.Module):
    def __init__(
        self,
        use_focal: bool = False,
        gamma: float = 1.5,
        alpha: float = 0.6,
        beta: float = 0.4,
        class_weights: list[float] = None,
        legacy: bool = False,
        ema_decay: float = 0.8,
        dynamic_power: float = 2.0,
        warmup_epochs: int = 3,
        rampup_epochs: int = 5,
        schedule_type: str = "linear",
        **kwargs,
    ):
        super().__init__()
        self.use_focal = use_focal
        self.legacy = legacy

        loss_cls = FocalLoss if self.use_focal else CrossEntropy
        self.ce = loss_cls(
            gamma=gamma,
            class_weights=class_weights,
            legacy=legacy,
            ema_decay=ema_decay,
            dynamic_power=dynamic_power,
            warmup_epochs=warmup_epochs,
            rampup_epochs=rampup_epochs,
            schedule_type=schedule_type,
            **kwargs,
        )

        if self.legacy:
            self.gdl = GeneralizedDice(**kwargs)
        else:
            self.gdl = TverskyLoss(alpha=alpha, beta=beta, **kwargs)

        self.s_ce = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.s_gdl = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        print(
            f"Initialized {self.__class__.__name__} (use_focal={use_focal}, legacy={legacy}, alpha={alpha}, beta={beta})"
        )

    def update_scheduled_weights(self, epoch: int) -> Tensor | None:
        if hasattr(self.ce, "update_scheduled_weights"):
            return self.ce.update_scheduled_weights(epoch)
        return None

    def update_dynamic_class_weights(
        self,
        val_dice_all_classes: Tensor,
        val_prec_all_classes: Tensor = None,
        val_rec_all_classes: Tensor = None,
        epoch: int = 0,
    ) -> Tensor | None:
        if hasattr(self.ce, "update_dynamic_class_weights"):
            return self.ce.update_dynamic_class_weights(
                val_dice_all_classes,
                val_prec_all_classes,
                val_rec_all_classes,
                epoch,
            )
        return None

    def forward(
        self, pred_softmax: Tensor, weak_target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        l_ce, _ = self.ce(pred_softmax, weak_target)
        l_gdl, _ = self.gdl(pred_softmax, weak_target)

        l_ce = l_ce.float()
        l_gdl = l_gdl.float()

        var_ce = 1.0 + F.softplus(self.s_ce)
        var_gdl = 1.0 + F.softplus(self.s_gdl)

        loss = (
            (0.5 / var_ce) * l_ce
            + 0.5 * torch.log(var_ce)
            + (0.5 / var_gdl) * l_gdl
            + 0.5 * torch.log(var_gdl)
        )

        return loss, l_ce.detach(), l_gdl.detach()
