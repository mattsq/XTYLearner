"""Joint Score-Based Factorisation model."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class ScoreNet(nn.Module):
    """Simple score network predicting continuous and discrete scores."""

    def __init__(self, d_x: int, d_y: int, k: int, hidden: int) -> None:
        super().__init__()
        self.t_embed = nn.Embedding(k, hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.trunk = nn.Sequential(
            nn.Linear(d_x + d_y + hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.score_head = nn.Linear(hidden, d_x + d_y)
        self.class_head = nn.Linear(hidden, k)

    def forward(
        self, xy: torch.Tensor, t_corrupt: torch.Tensor, tau: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.t_embed(t_corrupt)
        time_emb = self.time_mlp(tau)
        h = torch.cat([xy, t_emb, time_emb], dim=-1)
        h = self.trunk(h)
        score = self.score_head(h)
        logits = self.class_head(h)
        return score, logits


@register_model("jsbf")
class JSBF(nn.Module):
    """Joint Score-Based Factorisation.

    Parameters
    ----------
    d_x:
        Dimensionality of covariates ``X``.
    d_y:
        Dimensionality of outcomes ``Y``.
    hidden:
        Hidden dimension for the score network.
    timesteps:
        Number of diffusion steps.
    sigma_min, sigma_max:
        Minimum and maximum noise levels.
    cls_coef:
        Weight for the discrete denoising loss.
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        *,
        hidden: int = 256,
        timesteps: int = 1000,
        sigma_min: float = 0.002,
        sigma_max: float = 1.0,
        k: int = 2,
        cls_coef: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.hidden = hidden
        self.k = k
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cls_coef = cls_coef
        self.net = ScoreNet(d_x, d_y, k, hidden)

    # ----- diffusion utilities -----
    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def _q_sample_continuous(self, x0: torch.Tensor, t_idx: torch.Tensor):
        eps = torch.randn_like(x0)
        sig_t = self._sigma(t_idx.float() / self.timesteps).view(-1, 1)
        xt = x0 + sig_t * eps
        return xt, eps, sig_t

    def _q_sample_discrete(
        self, t0: torch.Tensor, t_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Corrupt a label following the D3PM process."""
        # Cosine corruption schedule following Austin et al. (2021)
        gamma = 0.5 * (
            1 - torch.cos(torch.pi * t_idx.float() / self.timesteps)
        ).view(-1)
        rand = torch.randint(0, self.k, t0.shape, device=t0.device)
        keep = torch.bernoulli(1 - gamma).bool()
        corrupted = torch.where(keep, t0, rand)
        mask = keep.logical_not()
        return corrupted, mask

    # ----- training objective -----
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        b = x.size(0)
        t_idx = torch.randint(0, self.timesteps, (b,), device=x.device)
        xy0 = torch.cat([x, y], dim=-1)
        xy_t, eps, sig_t = self._q_sample_continuous(xy0, t_idx)

        t_mask = t_obs != -1
        t_cln = t_obs.clone()
        if (~t_mask).any():
            rand_fill = torch.randint(
                0, self.k, (t_mask.logical_not().sum(),), device=x.device
            )
            t_cln[~t_mask] = rand_fill
        t_cln = t_cln.clamp_min(0)
        t_corrupt, _ = self._q_sample_discrete(t_cln, t_idx)
        tau = (t_idx.float() / self.timesteps).unsqueeze(-1)

        score_pred, logits_pred = self.net(xy_t, t_corrupt, tau)

        score_loss = (((score_pred + eps / sig_t) ** 2) * sig_t**2).mean()

        cls_per_row = F.cross_entropy(logits_pred, t_cln, reduction="none")
        cls_loss = cls_per_row[t_mask].mean()

        return score_loss + self.cls_coef * cls_loss

    # ----- simple sampler -----
    @torch.no_grad()
    def sample(self, n: int, extra_noise: float = 1.0):
        xy = torch.randn(
            n, self.d_x + self.d_y, device=self.net.score_head.weight.device
        )
        xy = xy * self.sigma_max * extra_noise
        t = torch.randint(0, self.k, (n,), device=xy.device)

        for t_idx in reversed(range(1, self.timesteps)):
            tau = torch.full((n, 1), t_idx / self.timesteps, device=xy.device)
            sig_t = self._sigma(tau)
            sig_prev = self._sigma(
                torch.tensor((t_idx - 1) / self.timesteps, device=xy.device)
            )
            step = sig_prev**2 - sig_t**2
            score, logits = self.net(xy, t, tau)
            xy = xy + step * score + step.sqrt() * torch.randn_like(xy)
            probs = F.softmax(logits, -1)
            t = torch.multinomial(probs, 1).squeeze(-1)

        x = xy[:, : self.d_x]
        y = xy[:, self.d_x :]
        return x.cpu(), y.cpu(), t.cpu()

    # ----- posterior utility -----
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return posterior ``p(t|x,y)`` using the classification head."""

        xy = torch.cat([x, y], dim=-1)
        tau = torch.zeros(x.size(0), 1, device=x.device)
        t_dummy = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        _, logits = self.net(xy, t_dummy, tau)
        return logits[:, : self.k].softmax(dim=-1)

    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
        return self._sample_conditional(x, t)

    @torch.no_grad()
    def _sample_conditional(
        self, x: torch.Tensor, t: torch.Tensor, extra_noise: float = 1.0
    ) -> torch.Tensor:
        n = x.size(0)
        y = torch.randn(n, self.d_y, device=x.device) * self.sigma_max * extra_noise
        xy = torch.cat([x, y], dim=-1)

        for t_idx in reversed(range(1, self.timesteps)):
            tau = torch.full((n, 1), t_idx / self.timesteps, device=x.device)
            sig_t = self._sigma(tau)
            sig_prev = self._sigma(
                torch.tensor((t_idx - 1) / self.timesteps, device=x.device)
            )
            step = sig_prev**2 - sig_t**2
            score, logits = self.net(xy, t, tau)
            xy[:, self.d_x :] += step * score[:, self.d_x :]
            xy[:, self.d_x :] += step.sqrt() * torch.randn_like(y)
            probs = F.softmax(logits, -1)
            t = torch.multinomial(probs, 1).squeeze(-1)
        return xy[:, self.d_x :]


__all__ = ["JSBF"]
