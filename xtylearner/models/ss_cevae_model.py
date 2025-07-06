"""Semi-supervised variant of CEVAE."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.functional import one_hot, softmax, gumbel_softmax
from torch.distributions import Normal

from .registry import register_model
from .layers import make_mlp


class EncoderZ(nn.Module):  # q(z | x,t,y)
    def __init__(
        self,
        d_x: int,
        k: int,
        d_y: int,
        d_z: int,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        dims = [d_x + k + d_y, *hidden_dims, d_z]
        self.mu = make_mlp(
            dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.log = make_mlp(
            dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(
        self, x: torch.Tensor, t_onehot: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.cat([x, t_onehot, y], -1)
        mu = self.mu(h)
        log = self.log(h).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * (0.5 * log).exp()
        return z, mu, log


class ClassifierT(nn.Module):  # q(t | x,y)
    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            [d_x + d_y, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], -1))


class DecoderX(nn.Module):  # p(x | z)
    def __init__(
        self,
        d_z: int,
        d_x: int,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
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


class DecoderT(nn.Module):  # p(t | z,x)
    def __init__(
        self,
        d_z: int,
        d_x: int,
        k: int,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            [d_z + d_x, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, x], -1))


class DecoderY(nn.Module):  # p(y | z,x,t)
    def __init__(
        self,
        d_z: int,
        d_x: int,
        k: int,
        d_y: int,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            [d_z + d_x + k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(
        self, z: torch.Tensor, x: torch.Tensor, t_onehot: torch.Tensor
    ) -> torch.Tensor:
        return self.net(torch.cat([z, x, t_onehot], -1))


@register_model("ss_cevae")
class SS_CEVAE(nn.Module):
    """Semi-supervised variant of CEVAE."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int = 2,
        d_z: int = 16,
        tau: float = 0.5,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        self.enc_z = EncoderZ(
            d_x,
            k,
            d_y,
            d_z,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.cls_t = ClassifierT(
            d_x,
            d_y,
            k,
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
            d_x,
            k,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.dec_y = DecoderY(
            d_z,
            d_x,
            k,
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
        lab = t_obs >= 0
        unlab = ~lab

        # ----- labelled branch -----
        t_lab = t_obs[lab]
        t1h_L = one_hot(t_lab, self.k).float()
        z_L, mu_L, logv_L = self.enc_z(x[lab], t1h_L, y[lab])

        log_px_L = Normal(self.dec_x(z_L), 1.0).log_prob(x[lab]).sum(-1)
        log_pt_L = -nn.CrossEntropyLoss(reduction="none")(
            self.dec_t(z_L, x[lab]), t_lab
        )
        log_py_L = Normal(self.dec_y(z_L, x[lab], t1h_L), 1.0).log_prob(y[lab]).sum(-1)
        kl_L = -0.5 * (1 + logv_L - mu_L.pow(2) - logv_L.exp()).sum(-1)
        elbo_L = (log_px_L + log_pt_L + log_py_L - kl_L).mean()

        # ----- unlabelled branch -----
        elbo_U = torch.tensor(0.0, device=x.device)
        if unlab.any():
            logits_q = self.cls_t(x[unlab], y[unlab])
            q_t = softmax(logits_q, -1)
            t_soft = gumbel_softmax(logits_q, tau=self.tau, hard=False)
            z_U, mu_U, logv_U = self.enc_z(x[unlab], t_soft, y[unlab])

            log_px_U = Normal(self.dec_x(z_U), 1.0).log_prob(x[unlab]).sum(-1)
            logits_pT = self.dec_t(z_U, x[unlab])
            log_pt_U = -(q_t * logits_pT.log_softmax(-1)).sum(-1)
            log_py_U = (
                Normal(self.dec_y(z_U, x[unlab], t_soft), 1.0)
                .log_prob(y[unlab])
                .sum(-1)
            )
            kl_U = -0.5 * (1 + logv_U - mu_U.pow(2) - logv_U.exp()).sum(-1)
            H_q = -(q_t * q_t.log()).sum(-1)
            elbo_U = (log_px_U + log_pt_U + log_py_U - kl_U + H_q).mean()

        ce_sup = 0.0
        if lab.any():
            ce_sup = nn.CrossEntropyLoss()(self.cls_t(x[lab], y[lab]), t_lab)

        return -(elbo_L + elbo_U) + ce_sup

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return posterior ``p(t|x,y)`` from the classifier ``cls_t``."""

        logits = self.cls_t(x, y)
        return logits.softmax(dim=-1)


__all__ = ["SS_CEVAE"]
