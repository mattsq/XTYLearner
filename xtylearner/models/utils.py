from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def ramp_up_sigmoid(epoch: int, ramp: int, max_val: float = 1.0) -> float:
    """Sigmoid-shaped ramp used for curriculum schedules.

    Parameters
    ----------
    epoch:
        Current training epoch.
    ramp:
        Duration of the ramp in epochs.
    max_val:
        Maximum value attained after ``ramp`` epochs.

    Returns
    -------
    float
        Scaling factor in ``[0, max_val]``.
    """
    t = min(epoch / ramp, 1.0)
    return max_val * math.exp(-5 * (1 - t) ** 2)


def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample ``z`` using the reparameterisation trick ``z = mu + sigma * eps``."""

    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_normal(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """KL divergence ``KL(q‖p)`` for diagonal-covariance Gaussians."""

    return 0.5 * (
        logvar_p
        - logvar_q
        + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
        - 1.0
    ).sum(1)


def gumbel_softmax(
    logits: torch.Tensor, tau: float, hard: bool = False
) -> torch.Tensor:
    """Sample from the Gumbel-Softmax distribution."""

    return F.gumbel_softmax(logits, tau=tau, hard=hard)


def log_categorical(t: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Log-probability of labels ``t`` under categorical ``logits``."""

    if t.dim() == 1 or t.size(1) == 1:
        t = F.one_hot(t.to(torch.long).view(-1), logits.size(-1)).float()
    return (t * F.log_softmax(logits, 1)).sum(1)


__all__ = [
    "ramp_up_sigmoid",
    "reparameterise",
    "kl_normal",
    "gumbel_softmax",
    "log_categorical",
]
