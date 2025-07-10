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


def make_conditional_flow(
    dim_xy: int, context_dim: int, n_layers: int = 6, hidden: int = 128
) -> Flow:
    """Real-NVP with context (one-hot T) injected in every coupling net."""
    transforms = []
    mask = torch.arange(dim_xy) % 2
    for _ in range(n_layers):
        transforms += [
            AffineCouplingTransform(
                mask,
                transform_net_create_fn=lambda c_in, c_out: ResidualNet(
                    c_in, c_out, hidden_features=hidden, context_features=context_dim
                ),
            ),
            ReversePermutation(dim_xy),
        ]
        mask = 1 - mask  # flip mask each block
    return Flow(CompositeTransform(transforms), StandardNormal([dim_xy]))


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
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.k = k
        self.flow = make_conditional_flow(
            d_x + d_y, k, n_layers=flow_layers, hidden=flow_hidden
        )
        self.clf = make_mlp(
            [d_x, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    # --------------------------------------------------------
    def forward(
        self, X: torch.Tensor, T: torch.Tensor, *, steps: int = 20, lr: float = 0.1
    ) -> torch.Tensor:
        """Approximate ``p(y|x,t)`` via gradient-based MAP estimation."""

        ctx = F.one_hot(T.to(torch.long), self.k).float()
        Y = torch.zeros(X.size(0), self.d_y, device=X.device, requires_grad=True)
        with torch.enable_grad():
            for _ in range(steps):
                xy = torch.cat([X, Y], dim=-1)
                ll = self.flow.log_prob(xy, context=ctx)
                grad = torch.autograd.grad(ll.sum(), Y, create_graph=False)[0]
                Y = (Y + lr * grad).detach().requires_grad_(True)
        return Y.detach()

    # ---------- log-likelihood for a minibatch --------------------------
    def loss(self, X, Y, T_obs):
        """T_obs = int in [0,K-1] for labelled rows, -1 for missing."""
        t_lab_mask = T_obs >= 0
        t_ulb_mask = ~t_lab_mask

        # ---- labelled part --------------------------------------------
        loss_lab = torch.tensor(0.0, device=X.device)
        if t_lab_mask.any():
            t_lab = T_obs[t_lab_mask]
            ctx_lab = torch.nn.functional.one_hot(t_lab, self.k).float()
            xy_lab = torch.cat([X[t_lab_mask], Y[t_lab_mask]], dim=-1)

            ll_flow = self.flow.log_prob(xy_lab, context=ctx_lab)
            ce_clf = cross_entropy_loss(self.clf(X[t_lab_mask]), t_lab)
            loss_lab = -(ll_flow.mean() - ce_clf)  # maximise ll_flow

        # ---- un-labelled part -----------------------------------------
        loss_ulb = torch.tensor(0.0, device=X.device)
        if t_ulb_mask.any():
            X_u, Y_u = X[t_ulb_mask], Y[t_ulb_mask]
            logits = self.clf(X_u)  # (B_u,K)
            log_p_t = logits.log_softmax(-1)  # log p_psi(T|X)

            # flow likelihood under each treatment (batch, K)
            xy_u = torch.cat([X_u, Y_u], dim=-1).unsqueeze(1)  # (B_u,1,D)
            ctx = (
                torch.eye(self.k, device=X.device)
                .unsqueeze(0)
                .repeat(X_u.size(0), 1, 1)
            )  # (B_u,K,K)
            ll = self.flow.log_prob(
                xy_u.expand(-1, self.k, -1).reshape(-1, xy_u.size(-1)),
                context=ctx.reshape(-1, self.k),
            ).view(
                X_u.size(0), self.k
            )  # (B_u,K)

            # log p(x,y) = logsumexp_t [ log p(t|x) + log p(x,y|t) ]
            lse = torch.logsumexp(log_p_t + ll, dim=-1)
            loss_ulb = -lse.mean()  # maximise log-evidence

        return loss_lab + loss_ulb

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Return posterior ``p(t|x,y)`` computed from the flow and classifier."""

        logits = self.clf(X).log_softmax(-1)  # log p_psi(t|x)
        xy = torch.cat([X, Y], dim=-1).unsqueeze(1)  # (B,1,D)
        ctx = torch.eye(self.k, device=X.device).unsqueeze(0).repeat(X.size(0), 1, 1)
        ll = self.flow.log_prob(
            xy.expand(-1, self.k, -1).reshape(-1, xy.size(-1)),
            context=ctx.reshape(-1, self.k),
        ).view(X.size(0), self.k)
        return (logits + ll).softmax(dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, X: torch.Tensor, T: int | torch.Tensor) -> torch.Tensor:
        """Return predicted outcome ``y`` for inputs ``x`` and treatment ``t``."""

        if isinstance(T, int):
            T = torch.full((X.size(0),), T, dtype=torch.long, device=X.device)
        return self.forward(X, T)


__all__ = ["MixtureOfFlows"]
