# cevae_ss.py  –  CPU-only, PyTorch ≥2.2
# Louizos et al. “Causal Effect Inference with Deep Latent-Variable Models”, NeurIPS 2017
# proceedings.neurips.cc

import torch
import torch.nn as nn

from .layers import make_mlp


# ------------------------------------------------------------


class EncoderZ(nn.Module):  # q(z | x,t,y)
    def __init__(
        self,
        d_x,
        k,
        d_y,
        d_z,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ):
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

    def forward(self, x, t_1h, y):
        h = torch.cat([x, t_1h, y], -1)
        mu = self.mu(h)
        log = self.log(h).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * (0.5 * log).exp()
        return z, mu, log


class ClassifierT(nn.Module):  # q(t | x,y)
    def __init__(
        self,
        d_x,
        d_y,
        k,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ):
        super().__init__()
        self.net = make_mlp(
            [d_x + d_y, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, x, y):  # logits
        return self.net(torch.cat([x, y], -1))


class DecoderX(nn.Module):  # p(x | z)
    def __init__(
        self,
        d_z,
        d_x,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ):
        super().__init__()
        self.net = make_mlp(
            [d_z, *hidden_dims, d_x],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, z):  # Gaussian mean
        return self.net(z)


class DecoderT(nn.Module):  # p(t | z,x)
    def __init__(
        self,
        d_z,
        d_x,
        k,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ):
        super().__init__()
        self.net = make_mlp(
            [d_z + d_x, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, z, x):  # logits
        return self.net(torch.cat([z, x], -1))


class DecoderY(nn.Module):  # p(y | z,x,t)
    def __init__(
        self,
        d_z,
        d_x,
        k,
        d_y,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ):
        super().__init__()
        self.net = make_mlp(
            [d_z + d_x + k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, z, x, t_1h):  # mean
        return self.net(torch.cat([z, x, t_1h], -1))
