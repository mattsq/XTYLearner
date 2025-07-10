"""Tiny toy dataset used for quick demonstrations.

This dataset is completely synthetic and generated on the fly.  It is
useful for unit tests or simple sanity checks.  The returned
``TensorDataset`` contains three tensors: covariates ``X`` of shape
``(n_samples, d_x)``, outcomes ``Y`` of shape ``(n_samples, 1)`` and a
binary treatment label ``T`` of shape ``(n_samples,)``.
"""

from __future__ import annotations


import numpy as np
import torch
from torch.utils.data import TensorDataset


def load_toy_dataset(
    n_samples: int = 100,
    d_x: int = 2,
    seed: int = 0,
) -> TensorDataset:
    """Generate a tiny synthetic dataset for quick experiments.

    Parameters
    ----------
    n_samples:
        Number of data points to generate.
    d_x:
        Dimensionality of the covariates.
    seed:
        Random seed used for reproducibility.

    Returns
    -------
    TensorDataset
        Dataset of shape ``(n_samples, d_x)``, outcome vector ``Y`` and
        binary treatment ``T``.
    """

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, d_x)).astype(np.float32)
    T = rng.integers(0, 2, size=n_samples).astype(np.int64)

    beta = rng.normal(size=d_x)
    Y = X @ beta + 0.5 * T + rng.normal(scale=0.1, size=n_samples)
    Y = Y.astype(np.float32)

    return TensorDataset(
        torch.from_numpy(X), torch.from_numpy(Y).unsqueeze(-1), torch.from_numpy(T)
    )


__all__ = ["load_toy_dataset"]
