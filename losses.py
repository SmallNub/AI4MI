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
from torch import Tensor

from utils import simplex, sset


class CrossEntropy:
    def __init__(self, class_weights: list[float] | Tensor = None, **kwargs):
        self.idk = kwargs["idk"]
        if class_weights is None:
            class_weights = [0.01, 5.0, 1.0, 5.0, 2.0]
        self.weights = torch.tensor(class_weights, dtype=torch.float32).to(kwargs.get("device", "cpu"))
        print(
            f"Initialized {self.__class__.__name__} with weights={self.weights.tolist()} and kwargs={kwargs}"
        )

    def __call__(
        self, pred_softmax: Tensor, weak_target: Tensor
    ) -> tuple[Tensor, list]:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        # Restrict to supervised classes
        p = pred_softmax[:, self.idk, ...]
        t = weak_target[:, self.idk, ...].float()
        w = self.weights[self.idk].view(1, len(self.idk), 1, 1)

        log_p = (p + 1e-6).log()

        # Multiply element-wise by weights and sum over all dimensions
        weighted_loss = -(t * log_p * w).sum()
        normalizer = (t * w).sum() + 1e-6

        loss = weighted_loss / normalizer
        return loss, []


class FocalLoss:
    def __init__(
        self, gamma: float = 0.5, class_weights: list[float] | Tensor = None, **kwargs
    ):
        self.idk = kwargs["idk"]
        self.gamma = gamma
        if class_weights is None:
            class_weights = [0.01, 5.0, 1.0, 5.0, 2.0]
        self.weights = torch.tensor(class_weights, dtype=torch.float32).to(kwargs.get("device", "cpu"))
        print(
            f"Initialized {self.__class__.__name__} with gamma={gamma}, weights={self.weights.tolist()}, and kwargs={kwargs}"
        )

    def __call__(
        self, pred_softmax: Tensor, weak_target: Tensor
    ) -> tuple[Tensor, list]:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        # Restrict to supervised classes
        p = pred_softmax[:, self.idk, ...]
        t = weak_target[:, self.idk, ...].float()
        w = self.weights[self.idk].view(1, len(self.idk), 1, 1)

        p_clamped = torch.clamp(p, min=1e-6, max=1.0 - 1e-6)

        log_p = p_clamped.log()
        log_1m_p = torch.log1p(-p_clamped)

        focal_weight = torch.exp(self.gamma * log_1m_p)

        focal_loss = -(t * focal_weight * log_p * w).sum()
        normalizer = (t * w).sum() + 1e-6

        loss = focal_loss / normalizer
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

        v_frac = volumes / torch.clamp(torch.sum(volumes), min=1e-6)
        weights = 1.0 / (torch.square(v_frac) + self.smooth)

        intersection = torch.sum(p * t, dim=(0, 2, 3))
        cardinality = torch.sum(p, dim=(0, 2, 3)) + volumes

        weights = weights[self.idk]
        intersection = intersection[self.idk]
        cardinality = cardinality[self.idk]

        gdl_num = torch.sum(weights * intersection)
        gdl_den = torch.sum(weights * cardinality) + self.eps

        gdl = 1.0 - (2.0 * gdl_num / gdl_den)
        return gdl, []


class CompoundLoss(nn.Module):
    def __init__(
        self,
        use_focal: bool = False,
        gamma: float = 0.5,
        class_weights: list[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.use_focal = use_focal

        if self.use_focal:
            self.ce = FocalLoss(gamma=gamma, class_weights=class_weights, **kwargs)
        else:
            self.ce = CrossEntropy(class_weights=class_weights, **kwargs)

        self.gdl = GeneralizedDice(**kwargs)

        self.s_ce = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.s_gdl = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        print(
            f"Initialized non-negative {self.__class__.__name__} (use_focal={use_focal})"
        )

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
