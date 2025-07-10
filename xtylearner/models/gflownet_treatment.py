"""Generative Flow Network for treatment assignment inference."""

from __future__ import annotations

import math
from typing import Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp

from .registry import register_model


class OutcomeModel(nn.Module):
    """Simple outcome likelihood model ``p_phi(y|x,t)``."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int,
        *,
        hidden_dims: tuple[int, ...] | list[int] = (128,),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.net = make_mlp(
            [d_x + k, *hidden_dims, d_y * 2],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_onehot = F.one_hot(t.to(torch.long), num_classes=self.k).float()
        h = self.net(torch.cat([x, t_onehot], dim=-1))
        mu, log_sigma = h.chunk(2, dim=-1)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
        return mu, sigma


class PolicyNet(nn.Module):
    """Policy network predicting logits ``log pi(t|x,y)``."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int,
        *,
        hidden_dims: tuple[int, ...] | list[int] = (128,),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            [d_x + d_y, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=-1))


class FlowNet(nn.Module):
    """Scalar flow value for the root state."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        *,
        hidden_dims: tuple[int, ...] | list[int] = (64,),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.fc = make_mlp(
            [d_x + d_y, *hidden_dims, 1],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([x, y], dim=-1)).squeeze(-1)


@register_model("gflownet_treatment")
class GFlowNetTreatment(nn.Module):
    """Minimal GFlowNet that samples treatments in proportion to outcome likelihood."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int = 2,
        *,
        outcome_hidden: tuple[int, ...] | list[int] = (128,),
        policy_hidden: tuple[int, ...] | list[int] = (128,),
        flow_hidden: tuple[int, ...] | list[int] = (64,),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.k = k
        self.outcome = OutcomeModel(
            d_x,
            d_y,
            k,
            hidden_dims=outcome_hidden,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.policy = PolicyNet(
            d_x,
            d_y,
            k,
            hidden_dims=policy_hidden,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.flow = FlowNet(
            d_x,
            d_y,
            hidden_dims=flow_hidden,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    # --------------------------------------------------------------
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        """Return trajectory balance + supervised outcome loss."""

        b = x.size(0)
        device = x.device

        ll_list = []
        for t_val in range(self.k):
            t_vec = torch.full((b,), t_val, dtype=torch.long, device=device)
            mu, s = self.outcome(x, t_vec)
            ll = -0.5 * (((y - mu) / s) ** 2 + 2 * s.log() + math.log(2 * math.pi)).sum(
                -1
            )
            ll_list.append(ll)
        ll_all = torch.stack(ll_list, dim=1)

        # sample action from policy or use observed treatment
        log_pi = self.policy(x, y)
        act = torch.where(
            t_obs == -1,
            torch.multinomial(F.softmax(log_pi, dim=-1), 1).squeeze(-1),
            t_obs,
        )
        log_pi_a = log_pi.gather(1, act.unsqueeze(-1)).squeeze(-1)

        # reward = outcome likelihood
        R = ll_all.gather(1, act.unsqueeze(-1)).squeeze(-1).exp()
        R = torch.clamp(R, min=1e-6)

        # trajectory balance loss (one-step case)
        F_root = self.flow(x, y)
        tb_loss = ((F_root + log_pi_a - R.log()) ** 2).mean()

        # supervised outcome loss for labelled rows
        ll_obs = ll_all.gather(1, t_obs.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        obs_mask = t_obs != -1
        outcome_loss = (
            -(ll_obs[obs_mask]).mean()
            if obs_mask.any()
            else torch.tensor(0.0, device=device)
        )

        return tb_loss + outcome_loss

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the mean of ``p(y|x,t)`` from the outcome model."""

        mu, _ = self.outcome(x, t)
        return mu

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return posterior ``p(t|x,y)`` from the policy network."""

        logits = self.policy(x, y)
        return logits.softmax(dim=-1)


__all__ = ["GFlowNetTreatment"]
