"""Synthetic dataset with a mix of labelled and unlabelled samples."""

from __future__ import annotations

import numpy as np
from torch.utils.data import TensorDataset

from .synthetic_dataset import load_synthetic_dataset


def load_mixed_synthetic_dataset(
    n_samples: int = 1000,
    d_x: int = 5,
    seed: int = 0,
    label_ratio: float = 0.5,
) -> TensorDataset:
    """Generate a synthetic dataset with partially missing treatments.

    Parameters
    ----------
    label_ratio:
        Fraction of samples with observed treatment labels.

    Returns
    -------
    TensorDataset
        Dataset ``(X, Y, T_obs)`` where unobserved ``T`` entries are ``-1``.
    """

    base = load_synthetic_dataset(n_samples=n_samples, d_x=d_x, seed=seed)
    X, Y, T = base.tensors
    rng = np.random.default_rng(seed + 1)
    unlab = rng.random(n_samples) >= label_ratio
    T_obs = T.clone()
    T_obs[unlab] = -1
    return TensorDataset(X, Y, T_obs)


__all__ = ["load_mixed_synthetic_dataset"]
