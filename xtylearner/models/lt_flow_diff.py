from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


def _sigma(tau: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Noise schedule used for the latent diffusion prior."""
    return sigma_min * (sigma_max / sigma_min) ** tau


class Coupling(nn.Module):
    """Simple RealNVP-style coupling layer conditioned on ``z``."""

    def __init__(self, dim: int, d_z: int, hidden: int) -> None:
        super().__init__()
        self.dim_a = dim - dim // 2
        self.dim_b = dim // 2
        self.scale = nn.Sequential(
            nn.Linear(self.dim_a + d_z, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.dim_b),
            nn.Tanh(),
        )
        self.shift = nn.Sequential(
            nn.Linear(self.dim_a + d_z, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.dim_b),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor, reverse: bool = False):
        xa, xb = torch.split(x, [self.dim_a, self.dim_b], dim=-1)
        h = torch.cat([xa, z], dim=-1)
        s, t = self.scale(h), self.shift(h)
        if reverse:
            xb = (xb - t) * torch.exp(-s)
            logdet = (-s).sum(-1)
        else:
            xb = xb * torch.exp(s) + t
            logdet = s.sum(-1)
        out = torch.cat([xa, xb], dim=-1)
        return out, logdet


class CondFlow(nn.Module):
    """Stack of coupling layers modelling ``p(x,y|z)``."""

    def __init__(self, dim_xy: int, d_z: int, hidden: int, n_layers: int = 4) -> None:
        super().__init__()
        self.couplings = nn.ModuleList(
            [Coupling(dim_xy, d_z, hidden) for _ in range(n_layers)]
        )
        self.register_buffer("base_mu", torch.zeros(dim_xy))

    def forward(self, xy: torch.Tensor, z: torch.Tensor):
        logdet = 0.0
        h = xy
        for c in self.couplings:
            h, ld = c(h, z, reverse=False)
            logdet = logdet + ld
        return h, logdet

    def inverse(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = u
        for c in reversed(self.couplings):
            h, _ = c(h, z, reverse=True)
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
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.d_z = d_z
        self.k = k
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        dim_xy = d_x + d_y
        self.encoder = Encoder(dim_xy, d_z, hidden)
        self.flow = CondFlow(dim_xy, d_z, hidden)
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

        u, logdet = self.flow(xy, z)
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
            mse_all.append(((s_t + noise / sig) ** 2).mean(dim=-1))
        mse_all = torch.stack(mse_all, dim=1)

        obs = t_obs != -1
        if obs.any():
            w = torch.full_like(mse_all, 1 / self.k)
            w[obs] = F.one_hot(t_obs[obs], self.k).float()
        else:
            w = torch.full_like(mse_all, 1 / self.k)
        score_loss = (w * mse_all).sum(dim=1).mean()

        ce_loss = torch.tensor(0.0, device=device)
        if obs.any():
            logits = self.classifier(x, y)
            ce_loss = F.cross_entropy(logits[obs], t_obs[obs].clamp_min(0))

        recon = log_pxy.mean()
        kld = 0.5 * (mu.pow(2) + logv.exp() - 1 - logv).sum(-1).mean()
        return -(recon - kld) + score_loss + ce_loss

    # ----- sampler -----
    @torch.no_grad()
    def paired_sample(
        self, x: torch.Tensor, n_steps: int = 30
    ) -> tuple[torch.Tensor, ...]:
        b = x.size(0)
        z = torch.randn(b, self.d_z, device=x.device)
        u = torch.randn(b, self.k, self.d_x + self.d_y, device=x.device)
        y = []
        for t_val in range(self.k):
            xy = self.flow.inverse(u[:, t_val], z)
            y.append(xy[:, -self.d_y :])
        return tuple(y)

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
