"""CEVAE with latent treatment for partially-observed T."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .layers import make_mlp
from .registry import register_model


class CondDiagGaussian(nn.Module):
    """Conditional diagonal Gaussian ``q(z|h)`` with reparameterisation."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.mu_layer = nn.Linear(hidden, out_dim)
        self.logv_layer = nn.Linear(hidden, out_dim)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(h)
        mu = self.mu_layer(h)
        logv = self.logv_layer(h).clamp(-8, 8)
        std = (0.5 * logv).exp()
        z = mu + std * torch.randn_like(std)
        return z, mu, logv

    def mu(self, h: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = self.net(h)
        return self.mu_layer(h)


def sample_or_clamp(logits: torch.Tensor, t_obs: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from Gumbel-Softmax or clamp to observed one-hot."""
    b, k = logits.size()
    mask = t_obs >= 0
    if mask.any():
        t_onehot = F.one_hot(t_obs.clamp_min(0), k).float()
    else:
        t_onehot = torch.zeros(b, k, device=logits.device)
    t_soft = F.gumbel_softmax(logits, tau=tau, hard=False)
    t_soft = torch.where(mask.unsqueeze(-1), t_onehot, t_soft)
    log_q_t = F.log_softmax(logits, dim=1)
    log_q_t = (t_soft * log_q_t).sum(1)
    return t_soft, log_q_t


@register_model("cevae_m")
class CEVAE_M(nn.Module):
    """CEVAE with latent treatment for partially-observed T."""

    k: int  # number of treatment categories

    def __init__(self, d_x: int, d_y: int, k: int = 2, d_z: int = 16, hidden: int = 64, tau: float = 0.5) -> None:
        super().__init__()
        self.k, self.tau = k, tau
        self.d_y = d_y
        # encoders
        self.enc_logits_t = make_mlp([d_x + d_y, hidden, k])
        self.enc_z = CondDiagGaussian(d_x + d_y + k, hidden, d_z)
        # decoders
        self.dec_x = make_mlp([d_z, hidden, d_x])
        self.dec_t = make_mlp([d_z, hidden, k])
        self.dec_y = make_mlp([d_x + k + d_z, hidden, d_y])

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict outcome ``y`` from covariates ``x`` and treatment ``t``."""
        t_onehot = F.one_hot(t, self.k).float() if t.dim() == 1 else t
        zeros_y = torch.zeros(x.size(0), self.d_y, device=x.device)
        z = self.enc_z.mu(torch.cat([x, zeros_y, t_onehot], 1))
        return self.dec_y(torch.cat([x, t_onehot, z], 1))

    # --------------------------------------------------------------
    def loss(self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
        logits_t = self.enc_logits_t(torch.cat([x, y], 1))
        t_soft, log_q_t = sample_or_clamp(logits_t, t_obs, self.tau)

        z, mu, logv = self.enc_z(torch.cat([x, y, t_soft], 1))

        log_px = Normal(self.dec_x(z), 1.0).log_prob(x).sum(1)
        log_pt = (t_soft * F.log_softmax(self.dec_t(z), 1)).sum(1)
        log_py = Normal(self.dec_y(torch.cat([x, t_soft, z], 1)), 1.0).log_prob(y).sum(1)
        kl_z = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum(1)

        elbo = (log_px + log_pt + log_py - kl_z - log_q_t).mean()
        if (t_obs >= 0).any():
            ce_sup = F.cross_entropy(logits_t[t_obs >= 0], t_obs[t_obs >= 0])
        else:
            ce_sup = torch.tensor(0.0, device=x.device)
        return -(elbo) + ce_sup

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.enc_logits_t(torch.cat([x, y], 1)), 1)

    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_onehot = F.one_hot(t, self.k).float() if t.dim() == 1 else t
        zeros_y = torch.zeros(x.size(0), self.d_y, device=x.device)
        z = self.enc_z.mu(torch.cat([x, zeros_y, t_onehot], 1))
        return self.dec_y(torch.cat([x, t_onehot, z], 1))

    @torch.no_grad()
    def sample_counterfactual(self, x: torch.Tensor, y: torch.Tensor, t_prime: torch.Tensor) -> torch.Tensor:
        t_hat = self.predict_treatment_proba(x, y).argmax(1)
        z = self.enc_z.mu(torch.cat([x, y, F.one_hot(t_hat, self.k).float()], 1))
        y_cf = self.dec_y(torch.cat([x, F.one_hot(t_prime, self.k).float(), z], 1))
        return y_cf


__all__ = ["CEVAE_M"]
