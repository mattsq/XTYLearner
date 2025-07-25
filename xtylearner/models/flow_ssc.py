# flow_ssc.py
# Dinh et al. “Density Estimation using Real NVP”, ICLR 2017
# arxiv.org
# Papamakarios et al. “Masked Autoregressive Flow”, NeurIPS 2017
# arxiv.org
# Liao et al. “SSCFlow: Semi-Supervised Conditional Normalizing Flow”, KBS 2023 (proposes the labelled + unlabelled ELBO we used)
# sciencedirect.com
# Ardizzone et al. “Conditional Invertible Neural Networks”, CVPR 2019
# arxiv.org

import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows.distributions import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms import (
    CompositeTransform,
    ReversePermutation,
    AffineCouplingTransform,
)
from nflows.nn.nets import ResidualNet

from typing import Callable, Iterable

from .registry import register_model
from ..training.metrics import cross_entropy_loss
from .layers import make_mlp


def _make_realnvp_flow(
    dim: int, *, context_dim: int = 0, n_layers: int = 6, hidden: int = 128
) -> Flow:
    """Small RealNVP block used for X and conditional Y flows."""

    def net(c_in: int, c_out: int) -> nn.Module:
        ctx = context_dim if context_dim > 0 else None
        return ResidualNet(c_in, c_out, hidden_features=hidden, context_features=ctx)

    transforms = []
    mask = (torch.arange(dim) % 2).bool()
    for _ in range(n_layers):
        transforms.append(AffineCouplingTransform(mask, net))
        transforms.append(ReversePermutation(dim))
        mask = ~mask

    base = StandardNormal([dim])
    return Flow(CompositeTransform(transforms), base)


def make_conditional_flow_y(
    d_x: int, d_y: int, k: int, n_layers: int = 6, hidden: int = 128
) -> Flow:
    """Conditional Real-NVP modelling ``p(y|x,t)``."""

    return _make_realnvp_flow(
        d_y, context_dim=d_x + k, n_layers=n_layers, hidden=hidden
    )


def make_unconditional_flow_x(d_x: int, n_layers: int = 2, hidden: int = 64) -> Flow:
    """Simple unconditional flow for ``p(x)``."""

    return _make_realnvp_flow(d_x, n_layers=n_layers, hidden=hidden)


@register_model("flow_ssc")
class MixtureOfFlows(nn.Module):
    """
    * one conditional flow  p_theta(x,y | t)  (context = one-hot(t))
    * one classifier        p_psi(t | x)
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int,
        *,
        hidden_dims: Iterable[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        flow_layers: int = 6,
        flow_hidden: int = 128,
        gamma: float = 1.0,
        eval_samples: int = 100,
        noise_std: float = 0.0,
        regr_samples: int = 1,
        regr_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.k = k
        self.gamma = gamma
        self.eval_samples = eval_samples
        self.noise_std = noise_std
        self.regr_samples = regr_samples
        self.regr_weight = regr_weight

        self.flow_x = make_unconditional_flow_x(
            d_x, n_layers=max(1, flow_layers // 2), hidden=flow_hidden // 2
        )
        self.flow_y = make_conditional_flow_y(
            d_x, d_y, k, n_layers=flow_layers, hidden=flow_hidden
        )
        self.clf = make_mlp(
            [d_x, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    # --------------------------------------------------------
    def forward(self, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Return ``E[y|t]`` estimated from flow samples."""

        return self.predict_outcome(X, T)

    # ---------- log-likelihood for a minibatch --------------------------
    def loss(self, X, Y, T_obs):
        """T_obs = int in [0,K-1] for labelled rows, -1 for missing."""
        if self.noise_std > 0 and self.training:
            Y = Y + self.noise_std * torch.randn_like(Y)

        lp_x = self.flow_x.log_prob(X)

        t_lab_mask = T_obs >= 0
        t_ulb_mask = ~t_lab_mask

        # ---- labelled part --------------------------------------------
        loss_lab = torch.tensor(0.0, device=X.device)
        if t_lab_mask.any():
            x_l = X[t_lab_mask]
            y_l = Y[t_lab_mask]
            t_l = T_obs[t_lab_mask]

            t_oh = F.one_hot(t_l, self.k).float()
            ctx_l = torch.cat([x_l, t_oh], dim=-1)

            ll_y = self.flow_y.log_prob(y_l, context=ctx_l)
            ce_clf = cross_entropy_loss(self.clf(x_l), t_l)
            ll_x = lp_x[t_lab_mask]

            loss_lab = -(ll_x + ll_y).mean() + ce_clf

            if self.regr_weight > 0:
                y_samples = self.flow_y.sample(self.regr_samples, context=ctx_l)
                y_mean = y_samples.mean(0)
                mse = F.mse_loss(y_mean, y_l)
                loss_lab = loss_lab + self.regr_weight * mse

        # ---- un-labelled part -----------------------------------------
        loss_ulb = torch.tensor(0.0, device=X.device)
        if t_ulb_mask.any():
            X_u, Y_u = X[t_ulb_mask], Y[t_ulb_mask]
            lp_x_u = lp_x[t_ulb_mask]
            logits = self.clf(X_u)
            log_p_t = logits.log_softmax(-1)

            y_rep = Y_u.unsqueeze(1).expand(-1, self.k, -1)
            x_rep = X_u.unsqueeze(1).expand(-1, self.k, -1)
            ctx = torch.cat([
                x_rep.reshape(-1, self.d_x),
                torch.eye(self.k, device=X.device)
                .unsqueeze(0)
                .expand(X_u.size(0), self.k, self.k)
                .reshape(-1, self.k),
            ], dim=-1)

            ll_y = self.flow_y.log_prob(y_rep.reshape(-1, self.d_y), context=ctx)
            ll_y = ll_y.view(X_u.size(0), self.k)

            lse = torch.logsumexp(log_p_t + ll_y, dim=-1)
            loss_ulb = -(lp_x_u + lse).mean()

        return loss_lab + self.gamma * loss_ulb

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Return posterior ``p(t|x,y)`` computed from the flow and classifier."""
        logits = self.clf(X).log_softmax(-1)

        y_rep = Y.unsqueeze(1).expand(-1, self.k, -1)
        x_rep = X.unsqueeze(1).expand(-1, self.k, -1)
        ctx = torch.cat([
            x_rep.reshape(-1, self.d_x),
            torch.eye(self.k, device=X.device)
            .unsqueeze(0)
            .expand(X.size(0), self.k, self.k)
            .reshape(-1, self.k),
        ], dim=-1)

        ll_y = self.flow_y.log_prob(y_rep.reshape(-1, self.d_y), context=ctx)
        ll_y = ll_y.view(X.size(0), self.k)
        return (logits + ll_y).softmax(dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, X: torch.Tensor, T: int | torch.Tensor) -> torch.Tensor:
        """Return ``E[y|t]`` estimated from flow samples."""

        if isinstance(T, int):
            T = torch.full((X.size(0),), T, dtype=torch.long, device=X.device)
        ctx = torch.cat([X, F.one_hot(T, self.k).float()], dim=-1)
        n = self.eval_samples if not self.training else 1
        y_samples = self.flow_y.sample(n, context=ctx)
        return y_samples.mean(1)


__all__ = ["MixtureOfFlows"]
