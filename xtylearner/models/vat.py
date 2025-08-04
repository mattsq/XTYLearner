from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.metrics import cross_entropy_loss, mse_loss
from .layers import make_mlp
from .registry import register_model
from .utils import ramp_up_sigmoid


def _l2_normalise(d: torch.Tensor) -> torch.Tensor:
    """Normalise a tensor batch-wise in L2 norm."""
    flat = d.view(d.size(0), -1)
    norm = torch.norm(flat, dim=1, keepdim=True) + 1e-8
    return d / norm.view(-1, *([1] * (d.dim() - 1)))


def vat_loss(
    model: nn.Module,
    x: torch.Tensor,
    xi: float = 1e-6,
    eps: float = 2.5,
    n_power: int = 1,
) -> torch.Tensor:
    """Virtual adversarial loss for a batch of inputs."""

    with torch.no_grad():
        pred = F.softmax(model(x), dim=1)

    d = torch.randn_like(x)
    for _ in range(n_power):
        d = xi * _l2_normalise(d)
        d.requires_grad_()
        pred_hat = model(x + d)
        adv_dist = F.kl_div(F.log_softmax(pred_hat, dim=1), pred, reduction="batchmean")
        grad = torch.autograd.grad(adv_dist, d)[0]
        d = grad.detach()

    r_adv = eps * _l2_normalise(d)
    pred_hat = model(x + r_adv)
    loss = F.kl_div(F.log_softmax(pred_hat, dim=1), pred, reduction="batchmean")
    return loss


@register_model("vat")
class VAT_Model(nn.Module):
    """Outcome model with VAT-regularised treatment classifier."""

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
        residual: bool = False,
        eps: float = 2.5,
        xi: float = 1e-6,
        n_power: int = 1,
        lambda_max: float = 1.0,
        ramp: int = 30,
    ) -> None:
        super().__init__()
        self.k = k
        self.eps = eps
        self.xi = xi
        self.n_power = n_power
        self.lambda_max = lambda_max
        self.ramp = ramp
        self.step = 0

        self.outcome = make_mlp(
            [d_x + k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )
        self.classifier = make_mlp(
            [d_x + d_y, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
        elif t.dim() == 0:
            t = t.expand(x.size(0)).to(torch.long)
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))

    # --------------------------------------------------------------
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        t_use = t_obs.clamp_min(0)
        t_onehot = F.one_hot(t_use.to(torch.long), self.k).float()

        y_hat = self.outcome(torch.cat([x, t_onehot], dim=-1))
        logits_t = self.classifier(torch.cat([x, y], dim=-1))

        mask = t_obs >= 0
        loss = torch.tensor(0.0, device=x.device)
        if mask.any():
            loss = mse_loss(y_hat[mask], y[mask]) + cross_entropy_loss(
                logits_t[mask], t_use[mask]
            )

        if torch.is_grad_enabled():
            vat_inp = torch.cat([x, y], dim=-1)
            L_vat = vat_loss(
                self.classifier,
                vat_inp,
                xi=self.xi,
                eps=self.eps,
                n_power=self.n_power,
            )
            lam = ramp_up_sigmoid(self.step, self.ramp, self.lambda_max)
            self.step += 1
            loss = loss + lam * L_vat
        return loss

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(torch.cat([x, y], dim=-1))
        return logits.softmax(dim=-1)


__all__ = ["VAT_Model", "vat_loss"]
