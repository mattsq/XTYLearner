# m2_simple.py
# Kingma et al. “Semi-Supervised Learning with Deep Generative Models”, NeurIPS 2014
# arxiv.org

import torch
import torch.nn as nn

from .layers import make_mlp


# ------------------------------------------------------------
# 2. the model components
class EncoderZ(nn.Module):  # q_phi(z | x,t)
    def __init__(
        self,
        d_x,
        k,
        d_z,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ):
        super().__init__()
        dims = [d_x + k, *hidden_dims, d_z]
        self.net_mu = make_mlp(
            dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.net_log = make_mlp(
            dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, x, t_onehot):
        h = torch.cat([x, t_onehot], -1)
        mu = self.net_mu(h)
        log = self.net_log(h).clamp(-8, 8)
        std = (0.5 * log).exp()
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log  # re-par trick


class ClassifierT(nn.Module):  # q_phi(t | x,y)
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

    def forward(self, x, y):
        return self.net(torch.cat([x, y], -1))  # logits


class DecoderX(nn.Module):  # p_theta(x | z)
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

    def forward(self, z):
        return self.net(z)  # Gaussian mean


class DecoderT(nn.Module):  # p_theta(t | x,z)
    def __init__(
        self,
        d_x,
        d_z,
        k,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ):
        super().__init__()
        self.net = make_mlp(
            [d_x + d_z, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, x, z):
        return self.net(torch.cat([x, z], -1))  # logits


class DecoderY(nn.Module):  # p_theta(y | x,t,z)
    def __init__(
        self,
        d_x,
        k,
        d_z,
        d_y,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
    ):
        super().__init__()
        self.net = make_mlp(
            [d_x + k + d_z, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, x, t_onehot, z):
        return self.net(torch.cat([x, t_onehot, z], -1))  # mean
