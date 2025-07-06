from __future__ import annotations

import math
import torch
import torch.nn as nn

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

    def __init__(self, d_z: int, hidden: int = 128, embed_dim: int = 64) -> None:
        super().__init__()
        self.t_emb = nn.Embedding(2, embed_dim)
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
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.d_z = d_z
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        dim_xy = d_x + d_y
        self.encoder = Encoder(dim_xy, d_z, hidden)
        self.flow = CondFlow(dim_xy, d_z, hidden)
        self.score = ScoreNet(d_z, hidden)

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
        k = torch.randint(1, self.timesteps + 1, (b,), device=device)
        tau = k.float() / self.timesteps
        sig = self._sigma(tau).unsqueeze(-1)
        noise = torch.randn_like(z)
        z_tau = z + sig * noise

        s0 = self.score(z_tau, torch.zeros_like(t_obs), tau.unsqueeze(-1))
        s1 = self.score(z_tau, torch.ones_like(t_obs), tau.unsqueeze(-1))
        mse0 = ((s0 + noise / sig) ** 2).mean(dim=-1)
        mse1 = ((s1 + noise / sig) ** 2).mean(dim=-1)

        obs = t_obs != -1
        if obs.any():
            w0 = (~obs).float() * 0.5 + obs.float() * (1 - t_obs.float())
            w1 = (~obs).float() * 0.5 + obs.float() * t_obs.float()
        else:
            w0 = w1 = torch.full_like(mse0, 0.5)
        score_loss = (w0 * mse0 + w1 * mse1).mean()

        recon = log_pxy.mean()
        kld = 0.5 * (mu.pow(2) + logv.exp() - 1 - logv).sum(-1).mean()
        return -(recon - kld) + score_loss

    # ----- sampler -----
    @torch.no_grad()
    def paired_sample(
        self, x: torch.Tensor, n_steps: int = 30
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        z = torch.randn(b, self.d_z, device=x.device)
        u0 = torch.randn(b, self.d_x + self.d_y, device=x.device)
        u1 = torch.randn_like(u0)
        xy0 = self.flow.inverse(u0, z)
        xy1 = self.flow.inverse(u1, z)
        return xy0[:, -self.d_y :], xy1[:, -self.d_y :]


__all__ = ["LTFlowDiff"]
