"""Generative Flow Network for treatment assignment inference."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class OutcomeModel(nn.Module):
    """Simple outcome likelihood model ``p_phi(y|x,t)``."""

    def __init__(self, d_x: int, d_y: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x + 1, hidden), nn.ReLU(), nn.Linear(hidden, d_y * 2)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([x, t.unsqueeze(-1)], dim=-1))
        mu, log_sigma = h.chunk(2, dim=-1)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
        return mu, sigma


class PolicyNet(nn.Module):
    """Policy network predicting logits ``log pi(t|x,y)``."""

    def __init__(self, d_x: int, d_y: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x + d_y, hidden), nn.ReLU(), nn.Linear(hidden, 2)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=-1))


class FlowNet(nn.Module):
    """Scalar flow value for the root state."""

    def __init__(self, d_x: int, d_y: int, hidden: int = 64) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_x + d_y, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([x, y], dim=-1)).squeeze(-1)


@register_model("gflownet_treatment")
class GFlowNetTreatment(nn.Module):
    """Minimal GFlowNet that samples treatments in proportion to outcome likelihood."""

    def __init__(self, d_x: int, d_y: int) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.outcome = OutcomeModel(d_x, d_y)
        self.policy = PolicyNet(d_x, d_y)
        self.flow = FlowNet(d_x, d_y)

    # --------------------------------------------------------------
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        """Return trajectory balance + supervised outcome loss."""

        b = x.size(0)
        device = x.device

        # outcome likelihoods for both treatment choices
        t0 = torch.zeros(b, dtype=torch.float32, device=device)
        t1 = torch.ones(b, dtype=torch.float32, device=device)
        mu0, s0 = self.outcome(x, t0)
        mu1, s1 = self.outcome(x, t1)
        ll0 = -0.5 * (
            ((y - mu0) / s0) ** 2 + 2 * s0.log() + math.log(2 * math.pi)
        ).sum(-1)
        ll1 = -0.5 * (
            ((y - mu1) / s1) ** 2 + 2 * s1.log() + math.log(2 * math.pi)
        ).sum(-1)

        # sample action from policy or use observed treatment
        log_pi = self.policy(x, y)
        act = torch.where(
            t_obs == -1,
            torch.multinomial(F.softmax(log_pi, dim=-1), 1).squeeze(-1),
            t_obs,
        )
        log_pi_a = log_pi.gather(1, act.unsqueeze(-1)).squeeze(-1)

        # reward = outcome likelihood
        R = torch.where(act == 0, ll0.exp(), ll1.exp())
        R = torch.clamp(R, min=1e-6)

        # trajectory balance loss (one-step case)
        F_root = self.flow(x, y)
        tb_loss = ((F_root + log_pi_a - R.log()) ** 2).mean()

        # supervised outcome loss for labelled rows
        ll_obs = torch.where(
            t_obs == 0,
            ll0,
            torch.where(t_obs == 1, ll1, torch.zeros_like(ll0)),
        )
        obs_mask = t_obs != -1
        outcome_loss = (
            -(ll_obs[obs_mask]).mean() if obs_mask.any() else torch.tensor(0.0, device=device)
        )

        return tb_loss + outcome_loss

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return posterior ``p(t|x,y)`` from the policy network."""

        logits = self.policy(x, y)
        return logits.softmax(dim=-1)


__all__ = ["GFlowNetTreatment"]
