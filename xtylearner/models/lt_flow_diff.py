from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from nflows.transforms import RandomPermutation


def _sigma(tau: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Noise schedule used for the latent diffusion prior."""
    return sigma_min * (sigma_max / sigma_min) ** tau


class Coupling(nn.Module):
    """
    Conditional coupling layer modelling p(y | x, z, t).
    x is passed as context and never transformed.
    """

    def __init__(self, d_x: int, d_y: int, d_z: int, k: int, hidden: int) -> None:
        super().__init__()
        self.d_y = d_y
        self.k = k
        self.scale = nn.Sequential(
            nn.Linear(d_x + d_z + k, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_y),
            nn.Tanh(),
        )
        self.shift = nn.Sequential(
            nn.Linear(d_x + d_z + k, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_y),
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        reverse: bool = False,
    ):
        ctx = torch.cat([x, z, F.one_hot(t, self.k).float()], dim=-1)
        s, t_shift = self.scale(ctx), self.shift(ctx)
        if reverse:
            y_new = (y - t_shift) * torch.exp(-s)
            logdet = (-s).sum(-1)
        else:
            y_new = y * torch.exp(s) + t_shift
            logdet = s.sum(-1)
        return y_new, logdet


class CondFlow(nn.Module):
    """y-flow: p(y | x,z,t)"""

    def __init__(self, d_x: int, d_y: int, d_z: int, k: int, hidden: int, n_layers: int = 6):
        super().__init__()
        self.transforms = nn.ModuleList(
            sum([[RandomPermutation(d_y), Coupling(d_x, d_y, d_z, k, hidden)] for _ in range(n_layers)], [])
        )
        self.register_buffer("base_mu", torch.zeros(d_y))

    def forward(self, y: torch.Tensor, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        logdet = 0.0
        h = y
        for tr in self.transforms:
            if isinstance(tr, RandomPermutation):
                h, ld = tr(h)
            else:
                h, ld = tr(x, h, z, t, reverse=False)
            logdet = logdet + ld
        return h, logdet

    def inverse(self, u: torch.Tensor, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = u
        for tr in reversed(self.transforms):
            if isinstance(tr, RandomPermutation):
                h, _ = tr.inverse(h)
            else:
                h, _ = tr(x, h, z, t, reverse=True)
        return h


class Encoder(nn.Module):
    """Gaussian encoder ``q(z|x,y)``."""

    def __init__(self, dim_xy: int, d_z: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_xy, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, d_z)
        self.logv = nn.Linear(hidden, d_z)

    def forward(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(xy)
        return self.mu(h), self.logv(h).clamp(-5, 5)


class ScoreNet(nn.Module):
    """Score network predicting ``\nabla_z log p(z|t)``."""

    def __init__(
        self, d_z: int, k: int, hidden: int = 128, embed_dim: int = 64
    ) -> None:
        super().__init__()
        self.t_emb = nn.Embedding(k, embed_dim)
        self.time = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(d_z + embed_dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_z),
        )

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        h = torch.cat([z, self.t_emb(t), self.time(tau)], dim=-1)
        return self.trunk(h)


class Classifier(nn.Module):
    """Classifier head estimating ``p(t|x,y)``."""

    def __init__(self, d_x: int, d_y: int, k: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x + d_y, hidden),
            nn.ReLU(),
            nn.Linear(hidden, k),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=-1))


@register_model("lt_flow_diff")
class LTFlowDiff(nn.Module):
    """Latent-Treatment Flow Diffusion model."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        *,
        d_z: int = 4,
        hidden: int = 128,
        timesteps: int = 1000,
        sigma_min: float = 0.002,
        sigma_max: float = 1.0,
        k: int = 2,
        lambda_score: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.d_z = d_z
        self.k = k
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.lambda_score = lambda_score
        dim_xy = d_x + d_y
        self.encoder = Encoder(dim_xy, d_z, hidden)
        self.flow = CondFlow(d_x, d_y, d_z, k, hidden)
        self.score = ScoreNet(d_z, k, hidden)
        self.classifier = Classifier(d_x, d_y, k, hidden)

    # ----- utilities -----
    def _sigma(self, tau: torch.Tensor) -> torch.Tensor:
        return _sigma(tau, self.sigma_min, self.sigma_max)

    # ----- training objective -----
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        device = x.device
        xy = torch.cat([x, y], dim=-1)
        mu, logv = self.encoder(xy)
        std = (0.5 * logv).exp()
        eps = torch.randn_like(std)
        z = mu + std * eps

        obs_mask = t_obs != -1
        t_in = t_obs.clone()
        confident = torch.tensor([], device=device)
        idx_u = torch.tensor([], dtype=torch.long, device=device)
        probs_u = None
        if (~obs_mask).any():
            idx_u = (~obs_mask).nonzero(as_tuple=True)[0]
            with torch.no_grad():
                pseudo_logits = self.classifier(x[idx_u], y[idx_u])
                probs_u = pseudo_logits.softmax(-1)
                probs_u = (probs_u ** 2) / probs_u.sum(dim=-1, keepdim=True)
                confident = probs_u.max(-1).values > 0.7
            t_in[idx_u] = probs_u.argmax(dim=-1)
        u, logdet = self.flow(y, x, z, t_in.clamp_min(0))
        log_pxy = (
            -0.5 * u.pow(2).sum(-1) - 0.5 * u.size(1) * math.log(2 * math.pi) + logdet
        )

        b = x.size(0)
        t_idx = torch.randint(1, self.timesteps + 1, (b,), device=device)
        tau = t_idx.float() / self.timesteps
        sig = self._sigma(tau).unsqueeze(-1)
        noise = torch.randn_like(z)
        z_tau = z + sig * noise

        mse_all = []
        for t_val in range(self.k):
            s_t = self.score(z_tau, torch.full_like(t_obs, t_val), tau.unsqueeze(-1))
            weight = sig.squeeze() ** 2
            mse_all.append(weight * ((s_t + noise / sig) ** 2).mean(dim=-1))
        mse_all = torch.stack(mse_all, dim=1)

        w = torch.full_like(mse_all, 1 / self.k)
        if obs_mask.any():
            w[obs_mask] = F.one_hot(t_obs[obs_mask], self.k).float()
        if idx_u.numel() > 0 and confident.any():
            w[idx_u[confident]] = probs_u[confident]
        score_loss = (w * mse_all).sum(dim=1).mean()

        ce_loss = torch.tensor(0.0, device=device)
        logits = self.classifier(x, y)
        if obs_mask.any():
            ce_loss = F.cross_entropy(logits[obs_mask], t_obs[obs_mask].clamp_min(0))
        if idx_u.numel() > 0 and confident.any():
            ce_loss = ce_loss + F.cross_entropy(logits[idx_u[confident]], probs_u[confident].argmax(dim=-1))

        recon = log_pxy.mean()
        return -recon + self.lambda_score * score_loss + ce_loss

    # ----- sampler -----
    @torch.no_grad()
    def sample_z(self, t_val: int, batch_size: int, n_steps: int = 30, step_size: float = 0.02) -> torch.Tensor:
        z = torch.randn(batch_size, self.d_z, device=self.score.t_emb.weight.device)
        tau = torch.ones(batch_size, 1, device=z.device)
        for _ in range(n_steps):
            sig = self._sigma(tau)
            grad = self.score(z, torch.full((batch_size,), t_val, device=z.device, dtype=torch.long), tau)
            z = z + 0.5 * step_size * grad + torch.sqrt(torch.tensor(step_size, device=z.device)) * torch.randn_like(z)
            tau = tau - 1.0 / n_steps
        return z

    @torch.no_grad()
    def paired_sample(
        self, x: torch.Tensor, n_steps: int = 30
    ) -> tuple[torch.Tensor, ...]:
        b = x.size(0)
        y_all = []
        for t_val in range(self.k):
            z = self.sample_z(t_val, b, n_steps, step_size=0.02)
            eps = torch.randn(b, self.d_y, device=x.device)
            y_t = self.flow.inverse(eps, x, z, torch.full((b,), t_val, device=x.device))
            y_all.append(y_t)
        return tuple(y_all)

    # ----- posterior utility -----
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return posterior ``p(t|x,y)`` estimated by the classifier."""

        logits = self.classifier(x, y)
        return logits.softmax(dim=-1)

    @torch.no_grad()
    def predict_outcome(
        self, x: torch.Tensor, t: torch.Tensor, n_steps: int = 30
    ) -> torch.Tensor:
        """Generate outcome predictions using the flow-based sampler."""

        y_all = self.paired_sample(x, n_steps=n_steps)
        y_stack = torch.stack(y_all, dim=1)
        t_long = t.view(-1, 1, 1).long()
        return y_stack.gather(1, t_long.expand(-1, 1, self.d_y)).squeeze(1)


__all__ = ["LTFlowDiff"]
