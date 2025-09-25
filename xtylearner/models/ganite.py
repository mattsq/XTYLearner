"""Adversarial GANITE model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from .layers import make_mlp
from .registry import register_model
from ..training.metrics import mse_loss, cross_entropy_loss


@register_model("ganite")
class GANITE(nn.Module):
    """Binary-treatment GANITE (Yoon et al., 2018)."""

    k = 2

    def __init__(
        self,
        d_x: int,
        d_y: int = 1,
        *,
        k: int | None = None,
        hidden_dims: tuple[int, ...] | list[int] = (200, 200),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        if k is not None:
            self.k = k
        super().__init__()
        self.G_cf = make_mlp(
            [d_x + d_y + self.k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.G_ite = make_mlp(
            [d_x + self.k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.D_y = make_mlp(
            [d_x + d_y + self.k, *hidden_dims, 1],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.C_t = make_mlp(
            [d_x + d_y, *hidden_dims, self.k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict outcome for covariates ``x`` under treatment ``t``."""

        t1h = F.one_hot(t.to(torch.long), self.k).float()
        return self.G_ite(torch.cat([x, t1h], dim=-1))

    # --------------------------------------------------------------
    def _impute_treatment(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one-hot treatment, counterfactual one-hot and logits."""
        logits = self.C_t(torch.cat([x, y], dim=-1))
        t_pred = logits.argmax(-1)
        mask = t_obs >= 0
        t_use = torch.where(mask, t_obs, t_pred)
        t1h = F.one_hot(t_use.to(torch.long), self.k).float()
        t_cf = 1 - t_use
        t1h_cf = F.one_hot(t_cf.to(torch.long), self.k).float()
        return t1h, t1h_cf, logits

    # --------------------------------------------------------------
    def loss_G(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        t1h, t1h_cf, logits = self._impute_treatment(x, y, t_obs)

        y_cf_hat = self.G_cf(torch.cat([x, y, t1h], dim=-1))
        d_out = self.D_y(torch.cat([x, y_cf_hat, t1h_cf], dim=-1))
        loss_adv = F.binary_cross_entropy_with_logits(d_out, torch.ones_like(d_out))

        y_f_hat = self.G_ite(torch.cat([x, t1h], dim=-1))
        mask = t_obs >= 0
        loss_f = (
            mse_loss(y_f_hat[mask], y[mask])
            if mask.any()
            else torch.tensor(0.0, device=x.device)
        )
        loss_cls = (
            cross_entropy_loss(logits[mask], t_obs[mask])
            if mask.any()
            else torch.tensor(0.0, device=x.device)
        )
        loss_g = loss_adv + loss_f + loss_cls
        return {"loss_G": loss_g}

    # --------------------------------------------------------------
    def loss_D(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        t1h, t1h_cf, _ = self._impute_treatment(x, y, t_obs)
        mask = t_obs >= 0
        if not mask.any():
            return {"loss_D": torch.tensor(0.0, device=x.device)}

        y_cf_hat = self.G_cf(torch.cat([x, y, t1h], dim=-1)).detach()
        real = self.D_y(torch.cat([x[mask], y[mask], t1h[mask]], dim=-1))
        fake = self.D_y(torch.cat([x[mask], y_cf_hat[mask], t1h_cf[mask]], dim=-1))
        loss_real = F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
        loss_fake = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
        return {"loss_D": loss_real + loss_fake}

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(
        self, x: torch.Tensor | np.ndarray, t: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=torch.long)
        t1h = F.one_hot(t.to(torch.long), self.k).float()
        return self.G_ite(torch.cat([x, t1h], dim=-1))

    @torch.no_grad()
    def predict_treatment_proba(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return ``p(t|x,y)`` from the classifier network."""

        z = torch.cat([x, y], dim=-1) if y is not None else x
        logits = self.C_t(z)
        return logits.softmax(-1)

    # --------------------------------------------------------------
    def generator_parameters(self):
        """Yield parameters optimised by the generator step."""

        yield from self.G_cf.parameters()
        yield from self.G_ite.parameters()
        yield from self.C_t.parameters()

    # --------------------------------------------------------------
    def discriminator_parameters(self):
        """Yield parameters optimised by the discriminator step."""

        yield from self.D_y.parameters()


__all__ = ["GANITE"]
