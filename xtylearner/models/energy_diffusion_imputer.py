"""Energy-guided Discrete Diffusion Imputer (EG-DDI)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class ScoreNet(nn.Module):
    """Score network predicting logits for the reverse diffusion."""

    def __init__(self, d_x: int, d_y: int, hidden: int = 128) -> None:
        super().__init__()
        self.x_proj = nn.Linear(d_x, hidden)
        self.y_proj = nn.Linear(d_y, hidden)
        self.t_embed = nn.Embedding(2, hidden)
        self.time_fc = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, t_corrupt: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        h = torch.cat(
            [self.x_proj(x), self.y_proj(y), self.t_embed(t_corrupt), self.time_fc(tau)],
            dim=-1,
        )
        return self.trunk(h)


class EnergyNet(nn.Module):
    """Simple energy model over ``(x,y,t)``."""

    def __init__(self, d_x: int, d_y: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x + d_y, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
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
        lr: float = 2e-4,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.timesteps = timesteps
        self.score_net = ScoreNet(d_x, d_y, hidden)
        self.energy_net = EnergyNet(d_x, d_y, hidden)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # ----- utilities -----
    def _gamma(self, k: torch.Tensor) -> torch.Tensor:
        return k.float() / self.timesteps

    # ----- training objective -----
    def loss(self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        device = x.device
        k = torch.randint(1, self.timesteps + 1, (b,), device=device)
        tau = k.float().unsqueeze(-1) / self.timesteps
        gam = self._gamma(k).unsqueeze(-1)

        t_clean = t_obs.clone()
        missing = t_obs == -1
        if missing.any():
            t_clean[missing] = torch.randint(0, 2, (missing.sum(),), device=device)
        flip = torch.bernoulli(gam).bool().squeeze(-1)
        t_noisy = torch.where(flip, 1 - t_clean, t_clean)

        logits = self.score_net(x, y, t_noisy, tau)

        obs_mask = t_obs != -1
        loss_obs = torch.tensor(0.0, device=device)
        if obs_mask.any():
            loss_obs = F.cross_entropy(logits[obs_mask], t_obs[obs_mask])

        loss_unobs = torch.tensor(0.0, device=device)
        if missing.any():
            # compute corrupted labels for both candidate values so the score
            # network sees the appropriate diffusion state for each label
            flip_m = flip[missing]
            tau_m = tau[missing]
            t_noisy0 = torch.where(flip_m, torch.ones_like(flip_m, dtype=torch.long), torch.zeros_like(flip_m, dtype=torch.long))
            t_noisy1 = torch.where(flip_m, torch.zeros_like(flip_m, dtype=torch.long), torch.ones_like(flip_m, dtype=torch.long))

            logits0 = self.score_net(x[missing], y[missing], t_noisy0, tau_m)
            logits1 = self.score_net(x[missing], y[missing], t_noisy1, tau_m)

            ce0 = F.cross_entropy(logits0, torch.zeros_like(t_noisy0), reduction="none")
            ce1 = F.cross_entropy(logits1, torch.ones_like(t_noisy1), reduction="none")

            E = self.energy_net(x[missing], y[missing])
            w = F.softmax(-E, dim=-1)
            loss_unobs = (w[:, 0] * ce0 + w[:, 1] * ce1).mean()

        E_all = self.energy_net(x, y)
        t_pos = torch.where(obs_mask, t_obs, logits.argmax(dim=-1))
        E_pos = E_all.gather(1, t_pos.unsqueeze(-1)).squeeze(-1)
        cd_loss = (E_pos + torch.logsumexp(-E_all, dim=-1)).mean()

        return loss_obs + loss_unobs + cd_loss

    # ----- optimizer helper -----
    def step(self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
        loss = self.loss(x, y, t_obs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    # ----- simple sampler -----
    @torch.no_grad()
    def sample_T(self, x: torch.Tensor, y: torch.Tensor, steps: int = 30):
        b = x.size(0)
        t = torch.randint(0, 2, (b,), device=x.device)
        for k in reversed(range(1, steps + 1)):
            tau = torch.full((b, 1), k / steps, device=x.device)
            logits = self.score_net(x, y, t, tau)
            energy = self.energy_net(x, y)
            guided = logits - energy
            probs = F.softmax(guided, dim=-1)
            t = torch.multinomial(probs, 1).squeeze(-1)
        return t


__all__ = ["EnergyDiffusionImputer"]
