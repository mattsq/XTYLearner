"""Semi-supervised generative M2VAE model."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.functional import one_hot, softmax, gumbel_softmax
from torch.distributions import Normal

from .registry import register_model
from .m2_vae import (
    EncoderZ as M2EncoderZ,
    ClassifierT as M2ClassifierT,
    DecoderX as M2DecoderX,
    DecoderT as M2DecoderT,
    DecoderY as M2DecoderY,
)


@register_model("m2_vae")
class M2VAE(nn.Module):
    """Simplified implementation of the M2 model."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int,
        d_z: int = 16,
        tau: float = 0.5,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        self.enc_z = M2EncoderZ(
            d_x,
            k,
            d_z,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.cls_t = M2ClassifierT(
            d_x,
            d_y,
            k,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.dec_x = M2DecoderX(
            d_z,
            d_x,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.dec_t = M2DecoderT(
            d_x,
            d_z,
            k,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.dec_y = M2DecoderY(
            d_x,
            k,
            d_z,
            d_y,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.k = k
        self.tau = tau

    def elbo(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        labelled = t_obs >= 0
        unlabelled = ~labelled

        # ----- labelled branch -----
        t_lab = t_obs[labelled]
        t1h_L = one_hot(t_lab, self.k).float()
        z_L, mu_L, logv_L = self.enc_z(x[labelled], t1h_L)

        recon_x_L = Normal(self.dec_x(z_L), 1.0).log_prob(x[labelled]).sum(-1)
        logits_t_L = self.dec_t(x[labelled], z_L)
        logp_t_L = -nn.CrossEntropyLoss(reduction="none")(logits_t_L, t_lab)
        recon_y_L = (
            Normal(self.dec_y(x[labelled], t1h_L, z_L), 1.0)
            .log_prob(y[labelled])
            .sum(-1)
        )
        kl_L = -0.5 * (1 + logv_L - mu_L.pow(2) - logv_L.exp()).sum(-1)
        elbo_L = (recon_x_L + logp_t_L + recon_y_L - kl_L).mean()

        # ----- unlabelled branch -----
        elbo_U = torch.tensor(0.0, device=x.device)
        if unlabelled.any():
            logits_q = self.cls_t(x[unlabelled], y[unlabelled])
            q_t = softmax(logits_q, -1)
            t_soft = gumbel_softmax(logits_q, tau=self.tau, hard=False)
            z_U, mu_U, logv_U = self.enc_z(x[unlabelled], t_soft)

            recon_x_U = Normal(self.dec_x(z_U), 1.0).log_prob(x[unlabelled]).sum(-1)
            logits_t_U = self.dec_t(x[unlabelled], z_U)
            logp_t_U = -(q_t * logits_t_U.log_softmax(-1)).sum(-1)
            recon_y_U = (
                Normal(self.dec_y(x[unlabelled], t_soft, z_U), 1.0)
                .log_prob(y[unlabelled])
                .sum(-1)
            )
            kl_U = -0.5 * (1 + logv_U - mu_U.pow(2) - logv_U.exp()).sum(-1)
            elbo_U = (
                recon_x_U + recon_y_U - kl_U + logp_t_U + (-(q_t * q_t.log()).sum(-1))
            ).mean()

        ce_sup = 0.0
        if labelled.any():
            ce_sup = nn.CrossEntropyLoss()(self.cls_t(x[labelled], y[labelled]), t_lab)

        return -(elbo_L + elbo_U) + ce_sup


__all__ = ["M2VAE"]
