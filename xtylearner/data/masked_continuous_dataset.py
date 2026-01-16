"""Synthetic dataset with partially-masked continuous treatments."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset


def load_masked_continuous_dataset(
    n_samples: int = 1000,
    d_x: int = 5,
    seed: int = 0,
    label_ratio: float = 0.5,
) -> TensorDataset:
    """Generate a synthetic dataset with partially-masked continuous treatments.

    This dataset is specifically designed for benchmarking models that support
    continuous treatments (k=None) with partial treatment masking.

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

    Returns
    -------
    TensorDataset
        Dataset ``(X, Y, T_obs)`` where unobserved ``T`` entries are ``-1``.
        The treatment tensor has dtype ``torch.float32`` representing continuous
        treatments.
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.normal(size=(n_samples, d_x)).astype(np.float32)

    # Generate continuous treatments from a linear combination of covariates + noise
    # Treatment is in range approximately [0, 1]
    w_t = rng.normal(size=d_x)
    T_raw = X @ w_t + rng.normal(scale=0.3, size=n_samples)
    # Normalize to [0, 1] range
    T_min, T_max = T_raw.min(), T_raw.max()
    T = ((T_raw - T_min) / (T_max - T_min)).astype(np.float32)

    # Generate outcomes with treatment effect
    # Y depends on X and T with heterogeneous treatment effects
    w_y = rng.normal(size=d_x)
    # Base outcome from covariates
    Y_base = X @ w_y
    # Treatment effect (heterogeneous based on first covariate)
    treatment_effect = 2.0 * T * (1.0 + 0.5 * X[:, 0])
    # Final outcome with noise
    Y = (Y_base + treatment_effect + rng.normal(scale=0.1, size=n_samples)).astype(
        np.float32
    )

    # Mask some treatments
    rng_mask = np.random.default_rng(seed + 1)
    unlab = rng_mask.random(n_samples) >= label_ratio
    T_obs = T.copy()
    T_obs[unlab] = -1.0

    return TensorDataset(
        torch.from_numpy(X), torch.from_numpy(Y).unsqueeze(-1), torch.from_numpy(T_obs)
    )


__all__ = ["load_masked_continuous_dataset"]
