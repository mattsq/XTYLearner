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
        if d_in is None and d_x is None:
            raise TypeError("d_in or d_x must be specified")
        if d_x is not None:
            d_in = d_x
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


__all__ = ["CTMT"]
