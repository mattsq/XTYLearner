from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model
from .utils import sinusoidal_time_embed, UNet1D


@register_model("ctm_t")
class CTMT(nn.Module):
    """Consistency-Trajectory Model with a treatment head."""

    def __init__(
        self,
        d_in: int | None = None,
        d_treat: int = 2,
        hidden: int = 64,
        *,
        d_x: int | None = None,
        d_y: int | None = None,
        k: int | None = None,
    ) -> None:
        if d_in is None:
            if d_x is None or d_y is None:
                raise TypeError("d_in or both d_x and d_y must be specified")
            d_in = d_x + d_y + 1
        if k is not None:
            d_treat = k
        super().__init__()
        self.d_in = d_in
        self.d_treat = d_treat
        self.backbone = UNet1D(d_in, hidden)
        self.time_emb = sinusoidal_time_embed(hidden)
        self.delta_emb = sinusoidal_time_embed(hidden)
        self.recon = nn.Linear(hidden, d_in)
        self.propensity = nn.Linear(hidden, d_treat)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, delta: torch.Tensor):
        emb = self.time_emb(t) + self.delta_emb(delta)
        h = self.backbone(x_t, emb)
        x_hat = self.recon(h)
        t_logits = self.propensity(h.detach())
        return x_hat, t_logits

    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return :math:`p(t\mid x,y)` inferred by the propensity head."""
        zeros = torch.zeros(x.size(0), 1, device=x.device)
        x0 = torch.cat([x, y, zeros], dim=-1)
        _, logits = self.forward(x0, zeros, zeros)
        return logits.softmax(dim=-1)

    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict outcome ``y`` for covariates ``x`` under treatment ``t``."""
        zeros = torch.zeros(x.size(0), 1, device=x.device)
        x0 = torch.cat([x, zeros, t.unsqueeze(-1).float()], dim=-1)
        out, _ = self.forward(x0, zeros, zeros)
        return out[:, x.size(1) : x.size(1) + 1]


__all__ = ["CTMT"]
