from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def ramp_up_sigmoid(epoch: int, ramp: int, max_val: float = 1.0) -> float:
    """Smoothly increase a scaling factor during training.

    The function returns a value in ``[0, max_val]`` following the
    ``exp(-5(1 - t)^2)`` schedule used in many curriculum learning
    schemes, where ``t`` is clipped to ``epoch / ramp``.

    Parameters
    ----------
    epoch:
        Current training epoch.
    ramp:
        Duration of the ramp in epochs.
    max_val:
        Maximum scaling factor after ``ramp`` epochs.

    Returns
    -------
    float
        The scaling value for the given ``epoch``.
    """
    t = min(epoch / ramp, 1.0)
    return max_val * math.exp(-5 * (1 - t) ** 2)


def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Draw a differentiable sample from ``N(mu, exp(logvar))``.

    Parameters
    ----------
    mu:
        Mean of the Gaussian distribution.
    logvar:
        Log-variance of the Gaussian.

    Returns
    -------
    torch.Tensor
        Random sample ``z`` computed as ``mu + sigma * eps`` where
        ``sigma = exp(0.5 * logvar)`` and ``eps`` is standard normal noise.
    """

    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_normal(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between two diagonal Gaussian distributions.

    Computes ``KL(N(mu_q, exp(logvar_q)) || N(mu_p, exp(logvar_p)))`` using
    the closed form expression for Gaussians with diagonal covariance.

    Returns
    -------
    torch.Tensor
        One-dimensional tensor containing the KL divergence for each row of
        ``mu_q``.
    """

    return 0.5 * (
        logvar_p
        - logvar_q
        + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
        - 1.0
    ).sum(1)


def gumbel_softmax(
    logits: torch.Tensor, tau: float, hard: bool = False
) -> torch.Tensor:
    """Sample from the Gumbel-Softmax distribution.

    Parameters
    ----------
    logits:
        Unnormalised log-probabilities for each category.
    tau:
        Temperature controlling the peakiness of the softmax samples.
    hard:
        If ``True``, the returned tensor is discretised using the straight
        through estimator.

    Returns
    -------
    torch.Tensor
        A sample with the same shape as ``logits`` drawn from a relaxed
        categorical distribution.
    """

    return F.gumbel_softmax(logits, tau=tau, hard=hard)


def log_categorical(t: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Return ``log p(t|logits)`` for a categorical distribution.

    Parameters
    ----------
    t:
        Integer class labels or one-hot encodings.
    logits:
        Unnormalised log-probabilities for each class.

    Returns
    -------
    torch.Tensor
        Tensor of log-probabilities with ``t.shape[0]`` elements.
    """

    if t.dim() == 1 or t.size(1) == 1:
        t = F.one_hot(t.to(torch.long).view(-1), logits.size(-1)).float()
    return (t * F.log_softmax(logits, 1)).sum(1)


def sinusoidal_time_embed(dim: int):
    """Return a function embedding scalar timesteps with sinusoids."""

    def embed(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t_local = t.unsqueeze(-1)
        else:
            t_local = t
        half = dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t_local.device, dtype=t_local.dtype)
            * -(math.log(10000.0) / max(half - 1, 1))
        )
        args = t_local * freqs
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(t_local)], dim=-1)
        return emb

    return embed


class UNet1D(nn.Module):
    """Minimal 1D UNet-like MLP used for tabular inputs."""

    def __init__(self, d_in: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, emb], dim=-1)
        return self.net(h)


__all__ = [
    "ramp_up_sigmoid",
    "reparameterise",
    "kl_normal",
    "gumbel_softmax",
    "log_categorical",
    "sinusoidal_time_embed",
    "UNet1D",
]
