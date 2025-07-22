from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model
from .utils import ramp_up_sigmoid


@register_model("mean_teacher")
class MeanTeacher(nn.Module):
    """Mean Teacher model for semi-supervised treatment inference."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int = 2,
        *,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: callable | None = None,
        ema_decay: float = 0.99,
        ramp_up: int = 40,
        cons_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.outcome = make_mlp(
            [d_x + k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.student = make_mlp(
            [d_x + d_y, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.teacher = deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.ema_decay = ema_decay
        self.ramp_up = ramp_up
        self.cons_max = cons_max
        self.step = 0

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict outcome ``y`` from covariates ``x`` and treatment ``t``."""

        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))

    # --------------------------------------------------------------
    def _consistency_weight(self) -> float:
        return ramp_up_sigmoid(self.step, self.ramp_up, self.cons_max)

    # --------------------------------------------------------------
    def update_teacher(self) -> None:
        with torch.no_grad():
            alpha = self.ema_decay
            for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
                p_t.data.mul_(alpha).add_(p_s.data, alpha=1 - alpha)

    # --------------------------------------------------------------
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        inp = torch.cat([x, y], dim=-1)

        def noise(z: torch.Tensor) -> torch.Tensor:
            std = z.std(0, keepdim=True) + 1e-6
            return z + 0.05 * std * torch.randn_like(z)

        inp_s = noise(inp)
        inp_t = noise(inp)

        logits_s = self.student(inp_s)
        with torch.no_grad():
            logits_t = self.teacher(inp_t)

        labelled = t_obs >= 0
        loss = torch.tensor(0.0, device=x.device)

        if labelled.any():
            t_use = t_obs[labelled].clamp_min(0)
            t_onehot = F.one_hot(t_use.to(torch.long), self.k).float()
            y_hat = self.outcome(torch.cat([x[labelled], t_onehot], dim=-1))
            loss += F.mse_loss(y_hat, y[labelled])
            loss += F.cross_entropy(logits_s[labelled], t_use)

        L_cons = F.mse_loss(logits_s.softmax(dim=-1), logits_t.softmax(dim=-1))
        lam = self._consistency_weight()
        loss = loss + lam * L_cons

        self.update_teacher()
        self.step += 1
        return loss

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        """Return outcome predictions for covariates ``x`` and treatment ``t``."""

        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
        elif t.dim() == 0:
            t = t.expand(x.size(0)).to(torch.long)
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.teacher(torch.cat([x, y], dim=-1))
        return logits.softmax(dim=-1)


__all__ = ["MeanTeacher"]
