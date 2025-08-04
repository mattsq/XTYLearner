from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.metrics import cross_entropy_loss, mse_loss
from .layers import make_mlp
from .registry import register_model
from .vat import vat_loss


@register_model("cycle_vat")
class CycleVAT(nn.Module):
    """Combine Cycle consistency with VAT regularisation.

    Consists of three networks:
    - a treatment classifier operating on ``x`` with virtual adversarial
      training (VAT),
    - an outcome generator predicting ``y`` from ``x`` and treatment ``t``,
    - an inverse classifier reconstructing ``t`` from ``x`` and the predicted
      outcome ``y``.
    """

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
        gradnorm: bool = False,
        gradnorm_alpha: float = 1.5,
        gradnorm_lr: float = 0.025,
    ) -> None:
        super().__init__()
        self.k = k
        self.eps = eps
        self.xi = xi
        self.n_power = n_power
        self.gradnorm = gradnorm
        self.gradnorm_alpha = gradnorm_alpha
        self.gradnorm_lr = gradnorm_lr
        self._init_losses: torch.Tensor | None = None
        if self.gradnorm:
            self.loss_weights = nn.Parameter(torch.ones(3))

        self.outcome = make_mlp(
            [d_x + k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )
        self.f_classifier = make_mlp(
            [d_x, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )
        self.i_classifier = make_mlp(
            [d_x + d_y, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
        elif t.dim() == 0:
            t = t.expand(x.size(0)).to(torch.long)
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))

    # ------------------------------------------------------------------
    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_obs: torch.Tensor,
        λ_sup: float = 1.0,
        λ_cyc: float = 1.0,
        λ_vat: float = 1.0,
    ) -> torch.Tensor:
        t_use = t_obs.clamp_min(0).to(torch.long)
        t_onehot = F.one_hot(t_use, self.k).float()

        logits_t = self.f_classifier(x)
        y_hat = self.outcome(torch.cat([x, t_onehot], dim=-1))
        inv_logits_t = self.i_classifier(torch.cat([x, y_hat.detach()], dim=-1))

        label_mask = t_obs >= 0
        ce_sup = (
            cross_entropy_loss(logits_t[label_mask], t_use[label_mask])
            if label_mask.any()
            else torch.tensor(0.0, device=x.device)
        )
        cycle = (
            mse_loss(y_hat[label_mask], y[label_mask])
            + cross_entropy_loss(inv_logits_t[label_mask], t_use[label_mask])
            if label_mask.any()
            else torch.tensor(0.0, device=x.device)
        )
        L_vat = (
            vat_loss(
                self.f_classifier, x, xi=self.xi, eps=self.eps, n_power=self.n_power
            )
            if torch.is_grad_enabled()
            else torch.tensor(0.0, device=x.device)
        )

        if self.gradnorm and self.training and torch.is_grad_enabled():
            losses = torch.stack([ce_sup, cycle, L_vat])
            if self._init_losses is None:
                self._init_losses = losses.detach() + 1e-8

            # weighted loss (detach to avoid gradients on weights from main loss)
            loss = (self.loss_weights.detach() * losses).sum()

            # compute gradient norms of each task
            params = [p for p in self.parameters() if p is not self.loss_weights]
            g_norms = []
            for w, L in zip(self.loss_weights, losses):
                g = torch.autograd.grad(
                    w * L,
                    params,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True,
                )
                g_norms.append(
                    torch.norm(torch.cat([gi.reshape(-1) for gi in g if gi is not None]))
                )
            g_norms = torch.stack(g_norms)

            with torch.no_grad():
                loss_ratios = losses.detach() / self._init_losses
                mean_ratio = loss_ratios.mean()
                target = g_norms.mean() * (loss_ratios / mean_ratio) ** self.gradnorm_alpha

            grad_loss = (g_norms - target.detach()).abs().sum()
            grad = torch.autograd.grad(grad_loss, self.loss_weights)[0]
            with torch.no_grad():
                self.loss_weights -= self.gradnorm_lr * grad
                self.loss_weights.data.clamp_(min=1e-3)
                self.loss_weights.data = 3 * self.loss_weights.data / self.loss_weights.data.sum()
            return loss

        return λ_sup * ce_sup + λ_cyc * cycle + λ_vat * L_vat

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        if y is None:
            raise ValueError("CycleVAT requires outcome `y` for treatment prediction")
        logits = self.i_classifier(torch.cat([x, y], dim=-1))
        return logits.softmax(dim=-1)


__all__ = ["CycleVAT"]
