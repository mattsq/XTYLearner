"""Factor-VAE+ generative model."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .registry import register_model
from .layers import make_mlp
from .utils import reparameterise as _reparameterise


class Encoder(nn.Module):
    """Gaussian encoder ``q(z|x,t,y)``."""

    def __init__(
        self,
        d_x: int,
        t_dim: int,
        d_y: int,
        d_z: int,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: Sequence[float] | float | None = None,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        dims = [d_x + t_dim + d_y, *hidden_dims]
        self.net = make_mlp(
            dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.mu = nn.Linear(dims[-1], d_z)
        self.logvar = nn.Linear(dims[-1], d_z)

    def forward(self, x: torch.Tensor, t_onehot: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x, t_onehot, y], dim=-1)
        h = self.net(h)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8, 8)
        return mu, logvar


class DecoderX(nn.Module):
    """Decoder ``p(x|z)`` predicting mean."""

    def __init__(
        self,
        d_z: int,
        d_x: int,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: Sequence[float] | float | None = None,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            [d_z, *hidden_dims, d_x],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DecoderT(nn.Module):
    """Decoder ``p(t|x,z)`` returning logits for each treatment."""

    def __init__(
        self,
        d_x: int,
        d_z: int,
        k: int,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: Sequence[float] | float | None = None,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        dims = [d_x + d_z, *hidden_dims]
        self.trunk = make_mlp(
            dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.heads = nn.ModuleList([nn.Linear(dims[-1], k)])

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> list[torch.Tensor]:
        h = self.trunk(torch.cat([x, z], dim=-1))
        return [head(h) for head in self.heads]


class DecoderY(nn.Module):
    """Decoder ``p(y|x,z,t)`` predicting mean."""

    def __init__(
        self,
        d_x: int,
        d_z: int,
        k: int,
        d_y: int,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: Sequence[float] | float | None = None,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            [d_x + d_z + k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor, t_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z, t_onehot], dim=-1))


@register_model("factor_vae_plus")
class FactorVAEPlus(nn.Module):
    """Factor-VAE+ model for multiple categorical treatments."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int,
        d_z: int = 16,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: Sequence[float] | float | None = None,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.cat_sizes = [k]
        self.z_dim = d_z

        self.encoder = Encoder(
            d_x,
            self.k,
            d_y,
            d_z,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.dec_x = DecoderX(
            d_z,
            d_x,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.dec_t = DecoderT(
            d_x,
            d_z,
            k,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.dec_y = DecoderY(
            d_x,
            d_z,
            self.k,
            d_y,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    # --------------------------------------------------------------
    def _one_hot(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t[:, 0]
        return F.one_hot(t.clamp_min(0), self.k).float()

    def elbo(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_oh = self._one_hot(t)
        mu, logvar = self.encoder(x, t_oh, y)
        z = _reparameterise(mu, logvar)

        recon_x = Normal(self.dec_x(z), 1.0).log_prob(x).sum(-1)
        logits_t = self.dec_t(x, z)[0]
        log_pt = x.new_zeros(x.size(0))
        target = t.squeeze(1) if t.dim() == 2 else t
        mask = target != -1
        if mask.any():
            log_pt[mask] = -F.cross_entropy(logits_t[mask], target[mask], reduction="none")
        recon_y = Normal(self.dec_y(x, z, t_oh), 1.0).log_prob(y).sum(-1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
        elbo = recon_x + log_pt + recon_y - kl
        return -elbo.mean()

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(x.size(0), self.z_dim, device=x.device)
        t_oh = self._one_hot(t)
        return self.dec_y(x, z, t_oh)

    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
        mu, logvar = self.encoder(
            x,
            torch.zeros(x.size(0), self.k, device=x.device),
            y,
        )
        z = _reparameterise(mu, logvar)
        logits = self.dec_t(x, z)
        return [F.softmax(logit, -1) for logit in logits]


__all__ = ["FactorVAEPlus"]
