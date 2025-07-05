"""Common metrics and loss functions used across trainers and models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error."""

    return F.mse_loss(pred, target)


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error."""

    return torch.mean(torch.abs(pred - target))


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root mean squared error."""

    return torch.sqrt(mse_loss(pred, target))


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross entropy for classification."""

    return F.cross_entropy(logits, target)


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Classification accuracy for ``logits`` compared with ``target``."""

    pred = logits.argmax(dim=-1)
    return (pred == target).float().mean()


__all__ = [
    "mse_loss",
    "mae_loss",
    "rmse_loss",
    "cross_entropy_loss",
    "accuracy",
]
