r"""Conditional normalising flow for ``p(y\mid x,t)``.

The outcome ``y`` is transformed by a stack of masked autoregressive blocks
while the treatment label ``t`` is injected as additional context alongside a
learned representation of ``x``.  Missing treatments can be marginalised out by
evaluating the likelihood for all classes and summing in the log domain.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    CompositeTransform,
    RandomPermutation,
)

from .registry import register_model
from .layers import make_mlp
from .utils import mmd


@register_model("cnflow")
class CNFlowModel(nn.Module):
    """Conditional flow estimating ``p(y|x,t)``.

    The flow acts only on the continuous outcome.  Treatment is encoded as part
    of the context together with an embedding of ``x``.  The negative
    log-likelihood objective marginalises over missing treatment labels when
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
        lambda_mmd: float = 0.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.d_y = d_y
        self.cond_net = make_mlp([d_x, hidden, hidden], activation=nn.ReLU)
        self.flow = self._build_conditional_flow(hidden, n_layers)
        self.eval_samples = eval_samples
        self.lambda_mmd = lambda_mmd

    # ------------------------------------------------------------
    def _build_conditional_flow(self, hidden: int, n_layers: int) -> Flow:
        """Return a conditional flow over ``y`` conditioned on ``x`` and ``t``."""

        transforms = []
        if self.d_y == 1:
            for _ in range(n_layers):
                transforms.append(
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                        features=self.d_y,
                        hidden_features=hidden,
                        context_features=hidden + self.k,
                        num_bins=8,
                        tails="linear",
                    )
                )
                transforms.append(RandomPermutation(features=self.d_y))
        else:
            for _ in range(n_layers):
                transforms.append(
                    MaskedAffineAutoregressiveTransform(
                        features=self.d_y,
                        hidden_features=hidden,
                        context_features=hidden + self.k,
                    )
                )
                transforms.append(RandomPermutation(features=self.d_y))
        transform = CompositeTransform(transforms)
        base = StandardNormal([self.d_y])
        return Flow(transform, base)

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_obs: torch.Tensor,
        propensity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the average negative log-likelihood for a mini-batch.

        Missing labels are denoted by ``-1`` in ``t_obs`` and are marginalised
        out when computing the likelihood.
        """

        y = y.float()
        if self.training:
            y = y + 0.01 * torch.randn_like(y)
        t_onehot = nn.functional.one_hot(t_obs.clamp_min(0), num_classes=self.k).float()
        ctx_x = self.cond_net(x)
        t_mask = t_obs != -1

        mmd_penalty = torch.tensor(0.0, device=x.device)
        if self.lambda_mmd > 0:
            vals = t_obs[t_mask].unique()
            pairs = []
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    idx_i = (t_obs == vals[i]) & t_mask
                    idx_j = (t_obs == vals[j]) & t_mask
                    if idx_i.any() and idx_j.any():
                        pairs.append(mmd(ctx_x[idx_i], ctx_x[idx_j]))
            if pairs:
                mmd_penalty = torch.stack(pairs).mean()

        logp_obs = torch.tensor(0.0, device=x.device)
        if t_mask.any():
            context = torch.cat([ctx_x[t_mask], t_onehot[t_mask]], dim=-1)
            logp_obs = self.flow.log_prob(y[t_mask], context)

        nll_obs = -logp_obs
        if propensity is not None and t_mask.any():
            w = 1.0 / propensity[t_mask, t_obs[t_mask]]
            nll_obs = -(w * logp_obs)
        nll = nll_obs.sum()
        if (~t_mask).any():
            y_m = y[~t_mask]
            ctx_x_m = ctx_x[~t_mask]
            n_m = ctx_x_m.size(0)
            logp_all = []
            for j in range(self.k):
                t_j = torch.zeros(n_m, self.k, device=x.device)
                t_j[:, j] = 1.0
                context = torch.cat([ctx_x_m, t_j], dim=-1)
                logp_all.append(self.flow.log_prob(y_m, context))
            logp_stack = torch.stack(logp_all, dim=-1)
            logp_miss = torch.logsumexp(logp_stack, dim=1)
            nll += -logp_miss.sum()
        return nll / x.size(0) + self.lambda_mmd * mmd_penalty

    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``E[y|x,t]`` estimated from flow samples."""

        return self.predict_outcome(x, t)

    # ------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``E[y|x,t]`` estimated from flow samples."""

        t_onehot = nn.functional.one_hot(t, num_classes=self.k).float()
        context = torch.cat([self.cond_net(x), t_onehot], dim=-1)
        n = self.eval_samples if not self.training else 1
        y_samples = self.flow.sample(n, context)
        return y_samples.mean(1)

    # ------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return posterior ``p(t|x,y)`` computed from the flow."""

        y = y.float()
        ctx_x = self.cond_net(x)
        logp_all = []
        for j in range(self.k):
            t_j = torch.zeros(len(x), self.k, device=x.device)
            t_j[:, j] = 1.0
            context = torch.cat([ctx_x, t_j], dim=-1)
            logp_all.append(self.flow.log_prob(y, context))
        logp_stack = torch.stack(logp_all, dim=-1)
        return logp_stack.softmax(dim=-1)

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
        context = torch.cat([self.cond_net(x), t_star_oh], dim=-1)
        return self.flow.sample(n, context)


__all__ = ["CNFlowModel"]
