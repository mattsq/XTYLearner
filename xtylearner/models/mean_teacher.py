from __future__ import annotations

from copy import deepcopy

import math
import torch
import torch.nn as nn

from .registry import register_model


@register_model("mean_teacher")
class MeanTeacher(nn.Module):
    """Simple mean teacher wrapper for semi-supervised classification."""

    def __init__(
        self,
        base_net_fn,
        num_classes: int,
        ema_decay: float = 0.99,
        ramp_up: int = 40,
        cons_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.student = base_net_fn(num_classes)
        self.teacher = deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.ema_decay = ema_decay
        self.ramp_up = ramp_up
        self.cons_max = cons_max
        self.step = 0  # global step counter

    # --------------------------------------------------------------
    def _consistency_weight(self, epoch: int) -> torch.Tensor:
        t = min(epoch / self.ramp_up, 1.0)
        return self.cons_max * math.exp(-5 * (1 - t) ** 2)

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, teacher: bool = False) -> torch.Tensor:
        net = self.teacher if teacher else self.student
        return net(x)

    # --------------------------------------------------------------
    def update_teacher(self) -> None:
        alpha = self.ema_decay
        for theta_s, theta_t in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            theta_t.data.mul_(alpha).add_(theta_s.data, alpha=1 - alpha)
