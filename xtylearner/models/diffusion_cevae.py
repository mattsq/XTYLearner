"""Diffusion-based CEVAE implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model


class EncoderU(nn.Module):
    """Gaussian encoder ``q(u|x,t,y)``."""

    def __init__(
        self,
        d_x: int,
        k: int,
        d_y: int,
        d_u: int,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        dims = [d_x + k + d_y, *hidden_dims]
        self.net = make_mlp(
            dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.mu = nn.Linear(dims[-1], d_u)
        self.log_var = nn.Linear(dims[-1], d_u)
        self.k = k

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        t1h = F.one_hot(t.clamp_min(0), self.k).float()
        mask = (t >= 0).unsqueeze(-1)
        t1h = torch.where(mask, t1h, torch.zeros_like(t1h))
        h = torch.cat([x, t1h, y], dim=-1)
        h = self.net(h)
        mu = self.mu(h)
        logv = self.log_var(h).clamp(-8, 8)
        return mu, logv


class DecoderX(nn.Module):
    """Decoder ``p(x|u)`` as Gaussian with unit variance."""

    def __init__(
        self,
        d_u: int,
        d_x: int,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            [d_u, *hidden_dims, d_x],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class DecoderT(nn.Module):
    """Decoder ``p(t|x,u)`` returning logits."""

    def __init__(
        self,
        d_x: int,
        d_u: int,
        k: int,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            [d_x + d_u, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, u], dim=-1))


class DecoderY(nn.Module):
    """Decoder ``p(y|x,t,u)`` predicting mean."""

    def __init__(
        self,
        d_x: int,
        d_u: int,
        k: int,
        d_y: int,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        self.net0 = make_mlp(
            [d_x + d_u, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.net1 = make_mlp(
            [d_x + d_u, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.k = k

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        h = torch.cat([x, u], dim=-1)
        y0 = self.net0(h)
        y1 = self.net1(h)
        t = t.view(-1, 1)
        return torch.where(t == 1, y1, y0)


class ScoreU(nn.Module):
    """Score network for the diffusion prior."""

    def __init__(self, d_u: int, hidden: int = 128) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.trunk = nn.Sequential(
            nn.Linear(d_u + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_u),
        )

    def forward(self, u: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # Ensure tau has correct dimensions for time_mlp
        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)
        elif tau.dim() > 2:
            tau = tau.view(-1, 1)
        t_emb = self.time_mlp(tau)
        h = torch.cat([u, t_emb], dim=-1)
        return self.trunk(h)


@register_model("diffusion_cevae")
class DiffusionCEVAE(nn.Module):
    """Diffusion-based CEVAE model."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int = 2,
        d_u: int = 16,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
        timesteps: int = 1000,
        sigma_min: float = 0.01,  # Increased from 0.001 for stability
        sigma_max: float = 1.0,
        lambda_score: float = 0.1,  # Reduced from 1.0 to prevent overwhelming other losses
    ) -> None:
        super().__init__()
        self.k = k
        self.d_u = d_u
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.lambda_score = lambda_score
        self.enc_u = EncoderU(
            d_x,
            k,
            d_y,
            d_u,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.dec_x = DecoderX(
            d_u,
            d_x,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.dec_t = DecoderT(
            d_x,
            d_u,
            k,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.dec_y = DecoderY(
            d_x,
            d_u,
            k,
            d_y,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.score_u = ScoreU(d_u, hidden=hidden_dims[0])

    # ----- diffusion utilities -----
    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        # Clamp t to prevent extreme values and ensure numerical stability
        t_clamped = torch.clamp(t, 0.0, 1.0)
        # Use log-space computation for better numerical stability
        ratio = self.sigma_max / self.sigma_min
        log_sigma = torch.log(torch.tensor(self.sigma_min)) + t_clamped * torch.log(torch.tensor(ratio))
        # Clamp sigma to prevent extremely small values that cause instability
        sigma = torch.clamp(log_sigma.exp(), self.sigma_min * 2, self.sigma_max)
        return sigma

    # ----- model loss -----
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        mu, logv = self.enc_u(x, t_obs, y)
        std = (0.5 * logv).exp()
        eps = torch.randn_like(std)
        u0 = mu + eps * std

        recon_x = F.mse_loss(self.dec_x(u0), x)
        logits_t = self.dec_t(x, u0)
        mask = t_obs >= 0
        if mask.any():
            recon_t = F.cross_entropy(logits_t[mask], t_obs[mask].clamp_min(0))
        else:
            recon_t = torch.tensor(0.0, device=x.device)
        t_in = torch.where(mask, t_obs.clamp_min(0), logits_t.argmax(dim=-1))
        recon_y = F.mse_loss(self.dec_y(x, t_in, u0), y)

        b = x.size(0)
        t_idx = torch.randint(1, self.timesteps + 1, (b,), device=x.device)
        tau = t_idx.float() / self.timesteps
        sig = self._sigma(tau).unsqueeze(-1)
        noise = torch.randn_like(u0)
        u_tau = u0 + sig * noise
        score_pred = self.score_u(u_tau, tau.unsqueeze(-1))
        
        # Add numerical stability to prevent explosion when sig is small
        eps = 1e-6
        sig_stable = sig + eps
        noise_over_sig = noise / sig_stable
        
        # Clamp the noise/sig ratio to prevent extreme values
        noise_over_sig = torch.clamp(noise_over_sig, -10.0, 10.0)
        
        score_loss = ((score_pred + noise_over_sig) ** 2).mean()

        return recon_x + recon_y + recon_t + self.lambda_score * score_loss

    # retain compatibility with GenerativeTrainer
    def elbo(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(x, y, t_obs)

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Approximate ``p(t|x,y)`` using the encoder and decoder."""

        b = x.size(0)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        t_dummy = torch.full((b,), -1, dtype=torch.long, device=x.device)
        mu, _ = self.enc_u(x, t_dummy, y)
        logits = self.dec_t(x, mu)
        return logits.softmax(dim=-1)

    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``p(y|x,t)`` approximated by the decoder."""

        u = torch.randn(x.size(0), self.d_u, device=x.device)
        return self.dec_y(x, t, u)


__all__ = ["DiffusionCEVAE"]
