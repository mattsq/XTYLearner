"""Utility functions for the NHEFS dataset."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

try:
    from causaldata import nhefs
except ImportError:
    nhefs = None


def load_nhefs_dataset(
    n_samples: int | None = None,
    seed: int = 0,
) -> TensorDataset:
    """Load the NHEFS dataset from the causaldata package.

    Parameters
    ----------
    n_samples:
        Number of samples to return. If None, returns the full dataset.
    seed:
        Random seed for sampling.

    Returns
    -------
    TensorDataset
        Dataset ``(X, Y, T)``.
    """
    if nhefs is None:
        raise ImportError("causaldata package not found. Please install it with `pip install causaldata`.")

    df = nhefs.load_pandas().data
    
    # For the purpose of this benchmark, we will consider 'qsmk' as the treatment
    # and 'wt82_71' as the outcome. All other variables will be covariates.
    
    treatment_col = "qsmk"
    outcome_col = "wt82_71"
    
    # Drop rows with missing values in outcome or treatment
    df = df.dropna(subset=[outcome_col, treatment_col])

    # Covariates are all columns except outcome and treatment
    covariate_cols = [col for col in df.columns if col not in [outcome_col, treatment_col]]
    
    # Convert to numpy arrays
    X = df[covariate_cols].values.astype(np.float32)
    Y = df[outcome_col].values.astype(np.float32)
    T = df[treatment_col].values.astype(np.int64)

    # Handle potential missing values in covariates by mean imputation
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])

    if n_samples is not None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X), n_samples, replace=False)
        X = X[indices]
        Y = Y[indices]
        T = T[indices]

    return TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(Y).unsqueeze(-1),
        torch.from_numpy(T),
    )

__all__ = ["load_nhefs_dataset"]
