"""Variational Auto-encoder with Conditional Integrated Masking (VACIM)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .utils import reparameterise, kl_normal, gumbel_softmax, log_categorical


class MLP(nn.Sequential):
    """Simple multi-layer perceptron used for encoders/decoders."""

    def __init__(self, dims: list[int]) -> None:
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        super().__init__(*layers)


@register_model("vacim")
class VACIM(nn.Module):
    """CEVAE variant with an additional partial encoder and treatment guide."""

    k: int

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int,
        d_z: int = 20,
        hidden: int = 128,
        n_mc: int = 5,
        temp_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.k, self.n_mc = k, n_mc
        self.temperature = temp_init
        # inference networks
        self.enc_z_full = MLP([d_x + k + d_y, hidden, hidden, 2 * d_z])
        self.enc_z_part = MLP([d_x + d_y, hidden, hidden, 2 * d_z])
        self.enc_t = MLP([d_x + d_y + d_z, hidden, hidden, k])
        # generative networks
        self.prior_z = MLP([d_x, hidden, hidden, 2 * d_z])
        self.dec_t = MLP([d_x + d_z, hidden, hidden, k])
        self.dec_y = MLP([d_x + k + d_z, hidden, hidden, 2 * d_y])
        self.d_y, self.d_z = d_y, d_z

    # --------------------------------------------------------------
    def elbo_observed(
        self, x: torch.Tensor, t_onehot: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_mu, z_logvar = self.enc_z_full(torch.cat([x, t_onehot, y], 1)).chunk(2, 1)
        z = reparameterise(z_mu, z_logvar)
        pz_mu, pz_logvar = self.prior_z(x).chunk(2, 1)

        y_mu, y_logvar = self.dec_y(torch.cat([x, t_onehot, z], 1)).chunk(2, 1)
        log_py = -0.5 * ((y - y_mu) ** 2 / torch.exp(y_logvar) + y_logvar).sum(1)
        log_pt = log_categorical(t_onehot, self.dec_t(torch.cat([x, z], 1)))

        kl_z = kl_normal(z_mu, z_logvar, pz_mu, pz_logvar)
        return log_py + log_pt - kl_z, y_mu

    def elbo_missing(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_logvar = self.enc_z_part(torch.cat([x, y], 1)).chunk(2, 1)
        elbo = x.new_zeros(len(x))
        y_hat = x.new_zeros(len(x), self.d_y)
        t_prob = x.new_zeros(len(x), self.k)
        for _ in range(self.n_mc):
            z = reparameterise(z_mu, z_logvar)
            t_logits = self.enc_t(torch.cat([x, y, z], 1))
            t_soft = gumbel_softmax(t_logits, self.temperature, hard=False)
            zc_mu, zc_logvar = self.enc_z_full(torch.cat([x, t_soft, y], 1)).chunk(2, 1)
            zc = reparameterise(zc_mu, zc_logvar)
            pz_mu, pz_logvar = self.prior_z(x).chunk(2, 1)

            y_mu, y_logvar = self.dec_y(torch.cat([x, t_soft, zc], 1)).chunk(2, 1)
            log_py = -0.5 * ((y - y_mu) ** 2 / torch.exp(y_logvar) + y_logvar).sum(1)
            log_pt = (t_soft * F.log_softmax(self.dec_t(torch.cat([x, zc], 1)), 1)).sum(1)
            log_qt = (t_soft * F.log_softmax(t_logits, 1)).sum(1)
            kl_z = kl_normal(zc_mu, zc_logvar, pz_mu, pz_logvar)
            elbo += (log_py + log_pt - log_qt - kl_z) / self.n_mc
            y_hat += y_mu / self.n_mc
            t_prob += torch.softmax(t_logits, 1) / self.n_mc
        return elbo, y_hat, t_prob

    # --------------------------------------------------------------
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Compute negative ELBO for a batch."""

        if t is None:
            mask = torch.zeros(len(x), dtype=torch.bool, device=x.device)
            t_obs = None
        else:
            if t.dim() == 1:
                if t.is_floating_point():
                    mask = ~torch.isnan(t)
                else:
                    mask = t != -1
                t_obs = t[mask]
            else:
                if t.is_floating_point():
                    mask = ~torch.isnan(t[:, 0])
                else:
                    mask = t[:, 0] != -1
                t_obs = t[mask]
            if t_obs.dim() == 2 and t_obs.size(1) == 1:
                t_obs = t_obs.squeeze(1)

        elbo = x.new_zeros(len(x))
        y_pred = x.new_zeros(len(x), self.d_y)
        t_imp = x.new_zeros(len(x), self.k)

        if mask.any():
            t_onehot = F.one_hot(t_obs.to(torch.long), self.k).float()
            e, y_hat = self.elbo_observed(x[mask], t_onehot, y[mask])
            elbo[mask], y_pred[mask] = e, y_hat
        if (~mask).any():
            e, y_hat, t_prob = self.elbo_missing(x[~mask], y[~mask])
            elbo[~mask], y_pred[~mask], t_imp[~mask] = e, y_hat, t_prob

        self.temperature = max(0.5, self.temperature * 0.999)
        return {"loss": -elbo.mean(), "y_hat": y_pred, "t_prob": t_imp}

    # --------------------------------------------------------------
    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--n_mc", type=int, default=5)
        parser.add_argument("--temp_init", type=float, default=1.0)
        return parser


__all__ = ["VACIM"]
