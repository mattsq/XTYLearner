from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model
from .fixmatch_tabular import fixmatch_unsup_loss


@register_model("fixmatch")
class FixMatch(nn.Module):
    """Outcome model with FixMatch-regularised treatment classifier."""

    def __init__(
        self,
        d_x: int = 1,
        d_y: int = 1,
        k: int = 2,
        *,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: callable | None = None,
        tau: float = 0.95,
        lambda_u: float = 1.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.tau = tau
        self.lambda_u = lambda_u
        self.outcome = make_mlp(
            [d_x + k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.classifier = make_mlp(
            [d_x + d_y, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict outcome ``y`` from covariates ``x`` and treatment ``t``."""

        t1h = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t1h], dim=-1))

    # --------------------------------------------------------------
    def loss(self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
        """Supervised loss with FixMatch regularisation."""

        labelled = t_obs >= 0
        t_use = t_obs.clamp_min(0)
        t1h = F.one_hot(t_use.to(torch.long), self.k).float()

        y_hat = self.outcome(torch.cat([x, t1h], dim=-1))
        logits = self.classifier(torch.cat([x, y], dim=-1))

        loss = torch.tensor(0.0, device=x.device)
        if labelled.any():
            loss = F.mse_loss(y_hat[labelled], y[labelled]) + F.cross_entropy(
                logits[labelled], t_use[labelled]
            )

        if (~labelled).any():
            x_u = torch.cat([x[~labelled], y[~labelled]], dim=-1)
            L_unsup = fixmatch_unsup_loss(self.classifier, x_u, tau=self.tau)
            loss = loss + self.lambda_u * L_unsup
        return loss

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(torch.cat([x, y], dim=-1))
        return logits.softmax(dim=-1)


__all__ = ["FixMatch"]
