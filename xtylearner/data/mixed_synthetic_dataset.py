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
    *,
    continuous_treatment: bool = False,
) -> TensorDataset:
    """Generate a synthetic dataset with partially missing treatments.

    Parameters
    ----------
    n_samples:
        Total number of samples to generate.
    d_x:
        Dimensionality of the covariates.
    seed:
        Seed controlling both dataset generation and label masking.
    label_ratio:
        Fraction of samples with observed treatment labels.
    continuous_treatment:
        If ``True`` keep the treatment array as ``np.float32`` and return a
        floating-point tensor. By default integer labels are returned.

    Returns
    -------
    TensorDataset
        Dataset ``(X, Y, T_obs)`` where unobserved ``T`` entries are ``-1``.
        When ``continuous_treatment=True`` the treatment tensor has dtype
        ``torch.float32``.
    """

    base = load_synthetic_dataset(
        n_samples=n_samples,
        d_x=d_x,
        seed=seed,
        continuous_treatment=continuous_treatment,
    )
    X, Y, T = base.tensors
    rng = np.random.default_rng(seed + 1)
    unlab = rng.random(n_samples) >= label_ratio
    T_obs = T.clone()
    T_obs[unlab] = -1
    return TensorDataset(X, Y, T_obs)


__all__ = ["load_mixed_synthetic_dataset"]
