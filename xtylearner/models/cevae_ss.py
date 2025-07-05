# cevae_ss.py  –  CPU-only, PyTorch ≥2.2
# Louizos et al. “Causal Effect Inference with Deep Latent-Variable Models”, NeurIPS 2017
# proceedings.neurips.cc

import torch
import torch.nn as nn


# ------------------------------------------------------------
def mlp(in_dim, out_dim, hid=128):
    return nn.Sequential(
        nn.Linear(in_dim, hid),
        nn.ReLU(),
        nn.Linear(hid, hid),
        nn.ReLU(),
        nn.Linear(hid, out_dim),
    )


class EncoderZ(nn.Module):  # q(z | x,t,y)
    def __init__(self, d_x, k, d_y, d_z):
        super().__init__()
        self.mu = mlp(d_x + k + d_y, d_z)
        self.log = mlp(d_x + k + d_y, d_z)

    def forward(self, x, t_1h, y):
        h = torch.cat([x, t_1h, y], -1)
        mu = self.mu(h)
        log = self.log(h).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * (0.5 * log).exp()
        return z, mu, log


class ClassifierT(nn.Module):  # q(t | x,y)
    def __init__(self, d_x, d_y, k):
        super().__init__()
        self.net = mlp(d_x + d_y, k)

    def forward(self, x, y):  # logits
        return self.net(torch.cat([x, y], -1))


class DecoderX(nn.Module):  # p(x | z)
    def __init__(self, d_z, d_x):
        super().__init__()
        self.net = mlp(d_z, d_x)

    def forward(self, z):  # Gaussian mean
        return self.net(z)


class DecoderT(nn.Module):  # p(t | z,x)
    def __init__(self, d_z, d_x, k):
        super().__init__()
        self.net = mlp(d_z + d_x, k)

    def forward(self, z, x):  # logits
        return self.net(torch.cat([z, x], -1))


class DecoderY(nn.Module):  # p(y | z,x,t)
    def __init__(self, d_z, d_x, k, d_y):
        super().__init__()
        self.net = mlp(d_z + d_x + k, d_y)

    def forward(self, z, x, t_1h):  # mean
        return self.net(torch.cat([z, x, t_1h], -1))
