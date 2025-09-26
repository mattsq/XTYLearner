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


class EncoderX(nn.Module):
    """Auxiliary encoder ``q(z|x)`` used for inference."""

    def __init__(
        self,
        d_x: int,
        d_z: int,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: Sequence[float] | float | None = None,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        dims = [d_x, *hidden_dims]
        self.net = make_mlp(
            dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.mu = nn.Linear(dims[-1], d_z)
        self.logvar = nn.Linear(dims[-1], d_z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
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
    """Decoder ``p(t|z)`` returning logits for each treatment."""

    def __init__(
        self,
        d_z: int,
        cat_sizes: Sequence[int],
        *,
        hidden_dims: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: Sequence[float] | float | None = None,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        dims = [d_z, *hidden_dims]
        self.trunk = make_mlp(
            dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.heads = nn.ModuleList([nn.Linear(dims[-1], size) for size in cat_sizes])

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        h = self.trunk(z)
        return [head(h) for head in self.heads]


class DecoderY(nn.Module):
    """Decoder ``p(y|x,z,t)`` predicting mean."""

    def __init__(
        self,
        d_x: int,
        d_z: int,
        t_dim: int,
        d_y: int,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: Sequence[float] | float | None = None,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            [d_x + d_z + t_dim, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor, t_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z, t_onehot], dim=-1))


class Discriminator(nn.Module):
    """Density-ratio discriminator used for total correlation."""

    def __init__(
        self,
        d_z: int,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: Sequence[float] | float | None = None,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            [d_z, *hidden_dims, 1],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


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
        cat_sizes: Sequence[int] | None = None,
        hidden_dims: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: Sequence[float] | float | None = None,
        norm_layer: type[nn.Module] | None = None,
        gamma: float = 1.0,
        disc_weight: float = 0.1,
        posterior_weight: float = 0.1,
        prediction_samples: int = 100,
    ) -> None:
        super().__init__()
        if cat_sizes is None:
            cat_sizes = [k]
        if len(cat_sizes) == 0:
            raise ValueError("cat_sizes must contain at least one entry")
        self.cat_sizes = list(cat_sizes)
        self.k = self.cat_sizes[0] if len(self.cat_sizes) == 1 else sum(self.cat_sizes)
        self.t_dim = sum(self.cat_sizes)
        self.z_dim = d_z
        self.gamma = gamma
        self.disc_weight = disc_weight
        self.posterior_weight = posterior_weight
        self.prediction_samples = prediction_samples

        self.encoder = Encoder(
            d_x,
            self.t_dim,
            d_y,
            d_z,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.encoder_x = EncoderX(
            d_x,
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
            d_z,
            self.cat_sizes,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.dec_y = DecoderY(
            d_x,
            d_z,
            self.t_dim,
            d_y,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.discriminator = Discriminator(
            d_z,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    # --------------------------------------------------------------
    def _one_hot(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.dim() == 2 and t.size(1) == 1 and len(self.cat_sizes) == 1:
            t = t
        elif t.dim() == 2 and t.size(1) != len(self.cat_sizes):
            raise ValueError(
                f"Expected treatment tensor with {len(self.cat_sizes)} columns, got {t.size(1)}"
            )
        elif t.dim() > 2:
            raise ValueError("Treatment tensor must be 1D or 2D")

        if t.dim() == 2 and t.size(1) == 1 and len(self.cat_sizes) > 1:
            raise ValueError(
                "Multiple categorical treatments require matching number of columns in `t`"
            )

        if t.dim() == 2:
            cols = [t[:, i] for i in range(t.size(1))]
        else:
            cols = [t.squeeze(-1)]

        onehots = []
        for idx, (col, size) in enumerate(zip(cols, self.cat_sizes)):
            col = col.long()
            mask = col >= 0
            oh = torch.zeros(col.size(0), size, device=col.device)
            if mask.any():
                oh[mask] = F.one_hot(col[mask], size).float()
            onehots.append(oh)
        return torch.cat(onehots, dim=-1)

    def _permute_dims(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(0) <= 1:
            return z.detach()
        permuted = []
        for dim in range(z.size(1)):
            perm = torch.randperm(z.size(0), device=z.device)
            permuted.append(z[perm, dim])
        return torch.stack(permuted, dim=1)

    def _sample_latent(
        self, mu: torch.Tensor, logvar: torch.Tensor, n_samples: int
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        if n_samples <= 1:
            return mu.unsqueeze(0)
        eps = torch.randn(n_samples, mu.size(0), mu.size(1), device=mu.device)
        return mu.unsqueeze(0) + eps * std.unsqueeze(0)

    def _kl_diag_gaussians(
        self,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor,
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
    ) -> torch.Tensor:
        var_p = logvar_p.exp()
        var_q = logvar_q.exp()
        return 0.5 * (
            (logvar_q - logvar_p)
            + (var_p + (mu_p - mu_q).pow(2)) / var_q
            - 1.0
        ).sum(dim=-1)

    def elbo(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_oh = self._one_hot(t)
        mu, logvar = self.encoder(x, t_oh, y)
        z = _reparameterise(mu, logvar)
        mu_x, logvar_x = self.encoder_x(x)

        recon_x = Normal(self.dec_x(z), 1.0).log_prob(x).sum(-1)
        logits_t = self.dec_t(z)
        log_pt = x.new_zeros(x.size(0))
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        for idx, logits in enumerate(logits_t):
            target = t[:, idx]
            mask = target != -1
            if mask.any():
                log_pt[mask] += -F.cross_entropy(
                    logits[mask], target[mask], reduction="none"
                )
        recon_y = Normal(self.dec_y(x, z, t_oh), 1.0).log_prob(y).sum(-1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
        elbo = recon_x + log_pt + recon_y - kl

        joint_logits = self.discriminator(z)
        tc_term = self.gamma * joint_logits.squeeze(-1)
        z_perm = self._permute_dims(z.detach())
        disc_logits_joint = self.discriminator(z.detach())
        disc_logits_perm = self.discriminator(z_perm)
        ones = torch.ones_like(disc_logits_joint)
        zeros = torch.zeros_like(disc_logits_perm)
        disc_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(disc_logits_joint, ones)
            + F.binary_cross_entropy_with_logits(disc_logits_perm, zeros)
        )

        kl_align = self._kl_diag_gaussians(mu, logvar, mu_x, logvar_x) + self._kl_diag_gaussians(
            mu_x, logvar_x, mu, logvar
        )

        loss = -((elbo + tc_term).mean())
        loss = loss + self.disc_weight * disc_loss + self.posterior_weight * kl_align.mean()
        return loss

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        n_samples: int | None = None,
    ) -> torch.Tensor:
        if n_samples is None:
            n_samples = self.prediction_samples
        mu_x, logvar_x = self.encoder_x(x)
        z_samples = self._sample_latent(mu_x, logvar_x, n_samples)
        t_oh = self._one_hot(t)
        x_rep = x.unsqueeze(0).expand(z_samples.size(0), -1, -1)
        t_rep = t_oh.unsqueeze(0).expand(z_samples.size(0), -1, -1)
        y_samples = self.dec_y(
            x_rep.reshape(-1, x.size(-1)),
            z_samples.reshape(-1, self.z_dim),
            t_rep.reshape(-1, self.t_dim),
        )
        y_samples = y_samples.view(z_samples.size(0), x.size(0), -1)
        return y_samples.mean(dim=0)

    @torch.no_grad()
    def predict_treatment_proba(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        *,
        n_samples: int | None = None,
    ) -> list[torch.Tensor]:
        if n_samples is None:
            n_samples = self.prediction_samples
        mu_x, logvar_x = self.encoder_x(x)
        z_samples = self._sample_latent(mu_x, logvar_x, n_samples)
        logits = self.dec_t(z_samples.reshape(-1, self.z_dim))
        probs = []
        for idx, logit in enumerate(logits):
            size = self.cat_sizes[idx]
            logit = logit.view(z_samples.size(0), x.size(0), size)
            probs.append(logit.softmax(-1).mean(dim=0))
        return probs


__all__ = ["FactorVAEPlus"]
