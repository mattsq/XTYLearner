# m2_simple.py
# Kingma et al. “Semi-Supervised Learning with Deep Generative Models”, NeurIPS 2014
# arxiv.org

import torch
import torch.nn as nn


# ------------------------------------------------------------
# 1. tiny helper MLP
def mlp(in_dim, out_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ------------------------------------------------------------
# 2. the model components
class EncoderZ(nn.Module):  # q_phi(z | x,t)
    def __init__(self, d_x, k, d_z):
        super().__init__()
        self.net_mu = mlp(d_x + k, d_z)
        self.net_log = mlp(d_x + k, d_z)

    def forward(self, x, t_onehot):
        h = torch.cat([x, t_onehot], -1)
        mu = self.net_mu(h)
        log = self.net_log(h).clamp(-8, 8)
        std = (0.5 * log).exp()
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log  # re-par trick


class ClassifierT(nn.Module):  # q_phi(t | x,y)
    def __init__(self, d_x, d_y, k):
        super().__init__()
        self.net = mlp(d_x + d_y, k)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], -1))  # logits


class DecoderX(nn.Module):  # p_theta(x | z)
    def __init__(self, d_z, d_x):
        super().__init__()
        self.net = mlp(d_z, d_x)

    def forward(self, z):
        return self.net(z)  # Gaussian mean


class DecoderT(nn.Module):  # p_theta(t | x,z)
    def __init__(self, d_x, d_z, k):
        super().__init__()
        self.net = mlp(d_x + d_z, k)

    def forward(self, x, z):
        return self.net(torch.cat([x, z], -1))  # logits


class DecoderY(nn.Module):  # p_theta(y | x,t,z)
    def __init__(self, d_x, k, d_z, d_y):
        super().__init__()
        self.net = mlp(d_x + k + d_z, d_y)

    def forward(self, x, t_onehot, z):
        return self.net(torch.cat([x, t_onehot, z], -1))  # mean
