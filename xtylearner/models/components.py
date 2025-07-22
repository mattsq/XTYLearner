"""Shared network components for causal factor models."""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import make_mlp


class VAE_T(nn.Module):
    """Simple VAE over multi-cause treatments with masked reconstruction."""

    def __init__(self, k_t: int, d_z: int, hidden: int = 64) -> None:
        super().__init__()
        self.enc_mu = make_mlp([k_t, hidden, hidden, d_z])
        self.enc_log = make_mlp([k_t, hidden, hidden, d_z])
        self.dec = make_mlp([d_z, hidden, hidden, k_t])
        self.d_z = d_z

    # ------------------------------------------------------------------
    def encode(self, t: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        """Encode observed treatments into a latent code.

        Parameters
        ----------
        t:
            Tensor of shape ``(n, k_t)`` containing possibly masked
            one-hot treatment labels. ``NaN`` entries are treated as
            missing values.
        sample:
            When ``True`` draw a sample ``z`` from the posterior
            ``q(z|t)``. If ``False`` the posterior mean is returned.

        Returns
        -------
        torch.Tensor
            Latent representation ``z`` with dimension ``d_z``.
        """

        t = t.float()
        t_filled = torch.nan_to_num(t, 0.0)
        mu = self.enc_mu(t_filled)
        logv = self.enc_log(t_filled).clamp(-8, 8)
        if sample:
            std = torch.exp(0.5 * logv)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z

    # ------------------------------------------------------------------
    def elbo(self, t: torch.Tensor) -> torch.Tensor:
        """Return the evidence lower bound for a batch of treatments."""
        t = t.float()
        mask = torch.isfinite(t)
        t_filled = torch.nan_to_num(t, 0.0)
        mu = self.enc_mu(t_filled)
        logv = self.enc_log(t_filled).clamp(-8, 8)
        std = torch.exp(0.5 * logv)
        z = mu + std * torch.randn_like(std)
        recon = self.dec(z)
        recon_loss = 0.5 * ((recon - t_filled) ** 2)
        recon_loss = (recon_loss * mask.float()).sum(-1).mean()
        kl = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum(-1).mean()
        return recon_loss + kl


class OutcomeNet(nn.Module):
    """Simple MLP outcome head used by :class:`DeconfounderCFM`."""

    def __init__(self, d_in: int, d_out: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = make_mlp([d_in, hidden, hidden, d_out])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Apply the MLP to ``x``."""
        return self.net(x)


__all__ = ["VAE_T", "OutcomeNet"]
