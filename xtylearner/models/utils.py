from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def ramp_up_sigmoid(epoch: int, ramp: int, max_val: float = 1.0) -> float:
    """Sigmoid ramp-up used for VAT and Mean Teacher baselines."""
    t = min(epoch / ramp, 1.0)
    return max_val * math.exp(-5 * (1 - t) ** 2)


def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterisation trick ``z = mu + sigma * eps``."""

    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_normal(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """KL divergence ``KL(q||p)`` for diagonal Gaussians."""

    return 0.5 * (
        logvar_p
        - logvar_q
        + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
        - 1.0
    ).sum(1)


def gumbel_softmax(
    logits: torch.Tensor, tau: float, hard: bool = False
) -> torch.Tensor:
    return F.gumbel_softmax(logits, tau=tau, hard=hard)


def log_categorical(t: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
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
