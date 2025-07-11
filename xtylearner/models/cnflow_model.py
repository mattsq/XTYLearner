"""Conditional normalising flow for joint modelling of ``(y, t) | x``.

The implementation uses masked autoregressive blocks from the ``nflows``
library to parameterise an invertible mapping between ``(y, t)`` and a base
Gaussian.  A small MLP produces conditioning features from ``x`` which are fed
to each flow layer.  Missing treatment labels can be marginalised out during
training by summing the likelihood over all possible treatments.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    MaskedAffineAutoregressiveTransform,
    CompositeTransform,
    IdentityTransform,
)

from .registry import register_model
from .layers import make_mlp


@register_model("cnflow")
class CNFlowModel(nn.Module):
    """Joint density model ``p(y, t | x)`` implemented with a conditional flow.

    The model concatenates the outcome ``y`` and treatment ``t`` into a single
    vector which is transformed via a stack of masked autoregressive layers.
    Conditioning features computed from ``x`` are injected into every layer.  A
    negative log-likelihood objective marginalises over missing treatments when
    necessary.
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int,
        hidden: int = 128,
        n_layers: int = 5,
        eval_samples: int = 100,
    ) -> None:
        super().__init__()
        self.k = k
        self.d_y = d_y
        self.cond_net = make_mlp([d_x, hidden, hidden], activation=nn.ReLU)
        self.flow = self._build_conditional_flow(hidden, n_layers)
        self.eval_samples = eval_samples

    # ------------------------------------------------------------
    def _build_conditional_flow(self, hidden: int, n_layers: int) -> Flow:
        """Return a conditional flow over ``(y, t)`` with context from ``x``."""

        transforms = []
        for _ in range(n_layers):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=self.d_y + self.k,
                    hidden_features=hidden,
                    context_features=hidden,
                )
            )
            transforms.append(IdentityTransform())
        transform = CompositeTransform(transforms)
        base = StandardNormal([self.d_y + self.k])
        return Flow(transform, base)

    # ------------------------------------------------------------
    def _joint_log_prob(
        self, x: torch.Tensor, y: torch.Tensor, t_onehot: torch.Tensor
    ) -> torch.Tensor:
        """Compute ``log p(y,t|x)`` for each sample."""

        context = self.cond_net(x)
        z = torch.cat([y, t_onehot], dim=-1)
        return self.flow.log_prob(z, context)

    # ------------------------------------------------------------
    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Return the average negative log-likelihood for a mini-batch.

        Missing labels are denoted by ``-1`` in ``t_obs`` and are marginalised
        out when computing the likelihood.
        """

        y = y.float()
        t_onehot = nn.functional.one_hot(t_obs.clamp_min(0), num_classes=self.k).float()

        t_mask = t_obs != -1

        logp_obs = torch.tensor(0.0, device=x.device)
        if t_mask.any():
            logp_obs = self._joint_log_prob(
                x[t_mask], y[t_mask], t_onehot[t_mask]
            )

        nll = -logp_obs.sum()
        if (~t_mask).any():
            x_m = x[~t_mask]
            y_m = y[~t_mask]
            all_t = torch.eye(self.k, device=x.device).repeat(len(x_m), 1)
            rep_x = x_m.repeat_interleave(self.k, dim=0)
            rep_y = y_m.repeat_interleave(self.k, dim=0)
            logp_all = self._joint_log_prob(rep_x, rep_y, all_t)
            logp_miss = torch.logsumexp(logp_all.view(len(x_m), self.k), dim=1)
            nll += -logp_miss.sum()
        return nll / x.size(0)

    # ------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``E[y|x,t]`` estimated from flow samples."""

        t_onehot = nn.functional.one_hot(t, num_classes=self.k).float()
        context = self.cond_net(x)
        n = self.eval_samples if not self.training else 1
        z = self.flow.sample(n, context)  # (B, n, D+k)
        context_exp = context.unsqueeze(1).expand(-1, n, -1)
        z[..., self.d_y :] = t_onehot.unsqueeze(1).expand(-1, n, -1)
        z_flat = z.reshape(-1, self.d_y + self.k)
        ctx_flat = context_exp.reshape(-1, context.size(-1))
        try:
            y_flat, _ = self.flow.inverse(z_flat, context=ctx_flat)
        except AttributeError:
            # fall back for older nflows
            y_flat, _ = self.flow._transform.inverse(z_flat, ctx_flat)
        y_samples = y_flat[..., : self.d_y].view(x.size(0), n, self.d_y)
        return y_samples.mean(1)

    # ------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return posterior ``p(t|x,y)`` computed from the flow."""

        y = y.float()
        all_t = torch.eye(self.k, device=x.device).repeat(len(x), 1)
        rep_x = x.repeat_interleave(self.k, dim=0)
        rep_y = y.repeat_interleave(self.k, dim=0)
        logp = self._joint_log_prob(rep_x, rep_y, all_t)
        return logp.view(len(x), self.k).softmax(dim=-1)

    # ------------------------------------------------------------
    @torch.no_grad()
    def potential_outcome(
        self, x: torch.Tensor, t_star: int | torch.Tensor, n: int | None = None
    ) -> torch.Tensor:
        """Draw samples from ``p(y|x,t_star)`` using the flow."""

        t_star_oh = nn.functional.one_hot(
            torch.as_tensor(t_star, device=x.device), self.k
        ).float()
        n = self.eval_samples if n is None else n
        context = self.cond_net(x)
        z = self.flow.sample(n, context)
        z[..., self.d_y :] = t_star_oh
        try:
            x_inv, _ = self.flow.inverse(z, context=context)
        except AttributeError:
            x_inv, _ = self.flow._transform.inverse(z, context)
        return x_inv[..., : self.d_y]


__all__ = ["CNFlowModel"]
