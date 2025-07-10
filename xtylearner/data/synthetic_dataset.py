"""Synthetic benchmark dataset used for experiments.

The dataset is generated procedurally using a simple structural model.
It provides covariates ``X``, outcomes ``Y`` and treatment labels
``T``.  Shapes are ``X`` of shape ``(n_samples, d_x)``, ``Y`` of shape
``(n_samples, 1)`` and ``T`` of shape ``(n_samples,)``.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_regression
import torch
from torch.utils.data import TensorDataset


def load_synthetic_dataset(
    n_samples: int = 1000,
    d_x: int = 5,
    seed: int = 0,
) -> TensorDataset:
    """Generate a simple synthetic benchmark dataset.

    Parameters
    ----------
    n_samples:
        Number of observations to generate.
    d_x:
        Dimensionality of the covariates ``X``.
    seed:
        Random seed controlling reproducibility.

    Returns
    -------
    TensorDataset
        Dataset ``(X, Y, T)`` with shapes ``(n_samples, d_x)``, ``(n_samples, 1)``
        and ``(n_samples,)``.
    """

    rng = np.random.default_rng(seed)
    X, _ = make_regression(
        n_samples=n_samples, n_features=d_x, noise=0.1, random_state=seed
    )
    X = X.astype(np.float32)

    w_t = rng.normal(size=d_x)
    logits_t = X @ w_t
    p_t = 1.0 / (1.0 + np.exp(-logits_t))
    T = rng.binomial(1, p_t).astype(np.int64)

    w_y = rng.normal(size=d_x)
    Y = X @ w_y + 2.0 * T + rng.normal(scale=0.1, size=n_samples)
    Y = Y.astype(np.float32)

    return TensorDataset(
        torch.from_numpy(X), torch.from_numpy(Y).unsqueeze(-1), torch.from_numpy(T)
    )


__all__ = ["load_synthetic_dataset"]
