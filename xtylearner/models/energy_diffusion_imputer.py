"""Energy-guided Discrete Diffusion Imputer (EG-DDI)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model


class ScoreNet(nn.Module):
    """Score network predicting logits for the reverse diffusion."""

    def __init__(self, d_x: int, d_y: int, k: int, hidden: int = 128) -> None:
        super().__init__()
        self.t_embed = nn.Embedding(k, hidden)
        self.x_proj = nn.Linear(d_x, hidden)
        self.y_proj = nn.Linear(d_y, hidden)
        self.time_fc = make_mlp([1, hidden, hidden], activation=nn.SiLU)
        self.trunk = make_mlp([hidden * 4, hidden, k], activation=nn.SiLU)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_corrupt: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.cat(
            [
                self.x_proj(x),
                self.y_proj(y),
                self.t_embed(t_corrupt),
                self.time_fc(tau),
            ],
            dim=-1,
        )
        return self.trunk(h)


class EnergyNet(nn.Module):
    """Simple energy model over ``(x,y,t)``."""

    def __init__(self, d_x: int, d_y: int, k: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = make_mlp(
            [d_x + d_y, hidden, hidden, k],
            activation=nn.ReLU,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=-1))


@register_model("eg_ddi")
class EnergyDiffusionImputer(nn.Module):
    """Energy-guided Discrete Diffusion Imputer."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        *,
        timesteps: int = 1000,
        hidden: int = 128,
        k: int = 2,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.k = k
        self.timesteps = timesteps
        self.score_net = ScoreNet(d_x, d_y, k, hidden)
        self.energy_net = EnergyNet(d_x, d_y, k, hidden)

    # --------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, *, steps: int = 20, lr: float = 0.1
    ) -> torch.Tensor:
        """Approximate ``p(y|x,t)`` via gradient-based energy minimisation."""

        y = torch.zeros(x.size(0), self.d_y, device=x.device, requires_grad=True)
        with torch.enable_grad():
            for _ in range(steps):
                e_all = self.energy_net(x, y)
                e = e_all.gather(1, t.view(-1, 1).clamp_min(0))
                grad = torch.autograd.grad(e.sum(), y, create_graph=False)[0]
                y = (y - lr * grad).detach().requires_grad_(True)
        return y.detach()

    # --------------------------------------------------------------
    def predict_outcome(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        steps: int = 20,
        lr: float = 0.1,
    ) -> torch.Tensor:
        """Convenience wrapper around :meth:`forward`.

        This method matches the interface expected by ``BaseTrainer`` for
        computing outcome metrics during training.
        """

        t = t.to(torch.long)
        return self.forward(x, t, steps=steps, lr=lr)

    # ----- diffusion utilities -----
    def _gamma(self, k: torch.Tensor) -> torch.Tensor:
        return k.float() / self.timesteps

    def _q_sample(self, t0: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Corrupt ``t0`` at step ``k`` by randomly replacing with random labels."""
        gam = self._gamma(k)
        flip = torch.bernoulli(gam).to(torch.bool)
        rand = torch.randint(0, self.k, t0.shape, device=t0.device)
        return torch.where(flip, rand, t0)

    # ----- training objective -----
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        b = x.size(0)
        device = x.device
        t_idx = torch.randint(1, self.timesteps + 1, (b,), device=device)
        tau = t_idx.float().unsqueeze(-1) / self.timesteps

        t_clean = t_obs.clone()
        missing = t_obs == -1
        if missing.any():
            t_clean[missing] = torch.randint(0, self.k, (missing.sum(),), device=device)
        t_corrupt = self._q_sample(t_clean, t_idx)

        logits = self.score_net(x, y, t_corrupt, tau)

        obs_mask = t_obs != -1
        loss_obs = torch.tensor(0.0, device=device)
        if obs_mask.any():
            loss_obs = F.cross_entropy(logits[obs_mask], t_obs[obs_mask])

        loss_unobs = torch.tensor(0.0, device=device)
        if missing.any():
            logits_m = logits[missing]
            log_probs = F.log_softmax(logits_m, dim=-1)
            E = self.energy_net(x[missing], y[missing])
            w = F.softmax(-E, dim=-1)
            loss_unobs = (w * (-log_probs)).sum(dim=-1).mean()

        E_all = self.energy_net(x, y)
        t_pos = torch.where(obs_mask, t_obs, logits.argmax(dim=-1))
        E_pos = E_all.gather(1, t_pos.unsqueeze(-1)).squeeze(-1)
        cd_loss = (E_pos + torch.logsumexp(-E_all, dim=-1)).mean()

        return loss_obs + loss_unobs + cd_loss

    # ----- simple sampler -----
    @torch.no_grad()
    def sample_T(
        self, x: torch.Tensor, y: torch.Tensor, steps: int = 30
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw labels ``t`` and return their final probabilities."""

        b = x.size(0)
        t = torch.randint(0, self.k, (b,), device=x.device)
        probs = None
        for step_idx in reversed(range(1, steps + 1)):
            tau = torch.full((b, 1), step_idx / steps, device=x.device)
            logits = self.score_net(x, y, t, tau)
            energy = self.energy_net(x, y)
            guided = logits - energy
            probs = F.softmax(guided, dim=-1)
            t = torch.multinomial(probs, 1).squeeze(-1)
        return t, probs

    @torch.no_grad()
    def predict_treatment_proba(
        self, x: torch.Tensor, y: torch.Tensor, steps: int = 30
    ) -> torch.Tensor:
        """Posterior ``p(t|x,y)`` estimated by the sampler."""

        _, probs = self.sample_T(x, y, steps)
        return probs


__all__ = ["EnergyDiffusionImputer"]
