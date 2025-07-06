from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class ScoreBridge(nn.Module):
    """Score network predicting ``\nabla_y log q(y_\tau | x,t)``."""

    def __init__(
        self, d_x: int, d_y: int, hidden: int = 256, embed_dim: int = 64
    ) -> None:
        super().__init__()
        self.t_embed = nn.Embedding(2, embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.x_proj = nn.Linear(d_x, embed_dim)
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim * 3 + d_y, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.score_head = nn.Linear(hidden, d_y)

    def forward(
        self, y_noisy: torch.Tensor, x: torch.Tensor, t: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.t_embed(t)
        tau_emb = self.time_mlp(tau)
        h = torch.cat([y_noisy, self.x_proj(x), t_emb, tau_emb], dim=-1)
        h = self.trunk(h)
        return self.score_head(h)


class Classifier(nn.Module):
    """Classifier head estimating ``p(t|x,y)``."""

    def __init__(self, d_x: int, d_y: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x + d_y, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=-1))


@register_model("bridge_diff")
class BridgeDiff(nn.Module):
    """Bridge-Diff Counterfactual model implementing a semi-supervised score loss."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        *,
        hidden: int = 256,
        timesteps: int = 1000,
        sigma_min: float = 0.002,
        sigma_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_y = d_y
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.score_net = ScoreBridge(d_x, d_y, hidden)
        self.classifier = Classifier(d_x, d_y)

    # ----- utilities -----
    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    # ----- training objective -----
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        b = x.size(0)
        device = x.device
        t_idx = torch.randint(1, self.timesteps + 1, (b,), device=device)
        tau = t_idx.float() / self.timesteps
        sig = self._sigma(tau).unsqueeze(-1)
        eps = torch.randn_like(y)
        y_noisy = y + sig * eps

        logits = self.classifier(x, y)
        p_post = F.softmax(logits.detach(), dim=-1)

        obs_mask = t_obs != -1
        loss_obs = torch.tensor(0.0, device=device)
        if obs_mask.any():
            s_obs = self.score_net(
                y_noisy[obs_mask],
                x[obs_mask],
                t_obs[obs_mask].clamp_min(0),
                tau[obs_mask].unsqueeze(-1),
            )
            loss_obs = ((s_obs + eps[obs_mask] / sig[obs_mask]) ** 2).mean()

        unobs_mask = obs_mask.logical_not()
        loss_unobs = torch.tensor(0.0, device=device)
        if unobs_mask.any():
            t0 = torch.zeros_like(t_obs[unobs_mask])
            t1 = torch.ones_like(t_obs[unobs_mask])
            s0 = self.score_net(
                y_noisy[unobs_mask],
                x[unobs_mask],
                t0,
                tau[unobs_mask].unsqueeze(-1),
            )
            s1 = self.score_net(
                y_noisy[unobs_mask],
                x[unobs_mask],
                t1,
                tau[unobs_mask].unsqueeze(-1),
            )
            mse0 = ((s0 + eps[unobs_mask] / sig[unobs_mask]) ** 2).mean(dim=-1)
            mse1 = ((s1 + eps[unobs_mask] / sig[unobs_mask]) ** 2).mean(dim=-1)
            w0 = p_post[unobs_mask, 0]
            w1 = p_post[unobs_mask, 1]
            loss_unobs = (w0 * mse0 + w1 * mse1).mean()

        ce_loss = torch.tensor(0.0, device=device)
        if obs_mask.any():
            ce_loss = F.cross_entropy(logits[obs_mask], t_obs[obs_mask].clamp_min(0))

        return loss_obs + loss_unobs + ce_loss

    # ----- sampler -----
    @torch.no_grad()
    def paired_sample(
        self, x: torch.Tensor, n_steps: int = 50
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one coupled draw ``(y0, y1)`` for each row in ``x``."""
        b = x.size(0)
        device = x.device
        y0 = torch.randn(b, self.d_y, device=device) * self.sigma_max
        y1 = y0.clone()

        for k in reversed(range(1, n_steps + 1)):
            tau = torch.full((b, 1), k / n_steps, device=device)
            sig = self._sigma(tau)
            score0 = self.score_net(
                y0,
                x,
                torch.zeros(b, dtype=torch.long, device=device),
                tau,
            )
            score1 = self.score_net(
                y1,
                x,
                torch.ones(b, dtype=torch.long, device=device),
                tau,
            )
            y0 = y0 + (sig**2) * score0
            y1 = y1 + (sig**2) * score1
            if k > 1:
                prev = tau - 1 / n_steps
                noise_scale = (sig**2 - self._sigma(prev).pow(2)).sqrt()
                noise = torch.randn_like(y0)
                y0 = y0 + noise_scale * noise
                y1 = y1 + noise_scale * noise
        return y0, y1

    # ----- posterior utility -----
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return posterior ``p(t|x,y)`` estimated by the classifier."""

        logits = self.classifier(x, y)
        return logits.softmax(dim=-1)


__all__ = ["BridgeDiff"]
