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
import torch.nn as nn
import torch.nn.functional as F
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

        return loss, []


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class GeneralizedDice:
    def __init__(self, **kwargs):
        self.idk = kwargs["idk"]
        self.eps = 1e-6
        self.smooth = 1e-2

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor) -> Tensor:
        p = pred_softmax.float()
        t = weak_target.float()

        volumes = torch.sum(t, dim=(0, 2, 3))

        v_frac = volumes / torch.clamp(torch.sum(volumes), min=1e-8)
        weights = 1.0 / (torch.square(v_frac) + self.smooth)

        intersection = torch.sum(p * t, dim=(0, 2, 3))
        cardinality = torch.sum(p, dim=(0, 2, 3)) + volumes

        weights = weights[self.idk]
        intersection = intersection[self.idk]
        cardinality = cardinality[self.idk]

        gdl_num = torch.sum(weights * intersection)
        gdl_den = torch.sum(weights * cardinality) + self.eps

        gdl = 1.0 - (2.0 * gdl_num / gdl_den)
        return gdl


class CompoundLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ce = CrossEntropy(**kwargs)
        self.gdl = GeneralizedDice(**kwargs)

        self.s_ce = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.s_gdl = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        print(f"Initialized non-negative {self.__class__.__name__}")

    def forward(self, pred_softmax: Tensor, weak_target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # 1. Unpack CrossEntropy return tuple (loss, loss_info)
        l_ce, _ = self.ce(pred_softmax, weak_target)
        l_gdl = self.gdl(pred_softmax, weak_target)

        # 2. Ensure floating point types match
        l_ce = l_ce.float()
        l_gdl = l_gdl.float()

        var_ce = 1.0 + F.softplus(self.s_ce)
        var_gdl = 1.0 + F.softplus(self.s_gdl)

        loss = (
            (0.5 / var_ce) * l_ce + 0.5 * torch.log(var_ce) +
            (0.5 / var_gdl) * l_gdl + 0.5 * torch.log(var_gdl)
        )

        return loss, l_ce.detach(), l_gdl.detach()
