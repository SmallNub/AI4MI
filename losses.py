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


import torch
from torch import Tensor, einsum

from utils import simplex, sset


class CrossEntropy:
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs["idk"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = -einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class GeneralizedDice:
    def __init__(self, **kwargs):
        self.idk = kwargs["idk"]
        self.eps = 1e-6
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor) -> Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        p = pred_softmax[:, self.idk, ...]
        t = weak_target[:, self.idk, ...].float()

        volumes = einsum("bkwh->bk", t)

        present = volumes > 0

        weights = 1.0 / (torch.clamp(volumes, min=1.0) ** 2)

        intersection = einsum("bkwh,bkwh->bk", p, t)
        cardinality = einsum("bkwh->bk", p) + volumes

        gdl_num = weights * intersection
        gdl_den = weights * cardinality + self.eps

        dice_per_class = (2.0 * gdl_num) / gdl_den
        loss_per_class = 1.0 - dice_per_class

        masked_loss = loss_per_class * present.float()

        num_present = present.sum()
        if num_present == 0:
            return torch.tensor(0.0, device=pred_softmax.device)

        return masked_loss.sum() / num_present


class CompoundLoss:
    def __init__(self, **kwargs):
        self.ce_weight = kwargs.get("ce_weight", 0.5)
        self.dice_weight = kwargs.get("dice_weight", 0.5)

        self.ce = CrossEntropy(**kwargs)
        self.gdl = GeneralizedDice(**kwargs)

        print(
            f"Initialized {self.__class__.__name__} with CE weight={self.ce_weight}, Dice weight={self.dice_weight}"
        )

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor) -> Tensor:
        ce_val = self.ce(pred_softmax, weak_target)
        gdl_val = self.gdl(pred_softmax, weak_target)

        return self.ce_weight * ce_val + self.dice_weight * gdl_val
