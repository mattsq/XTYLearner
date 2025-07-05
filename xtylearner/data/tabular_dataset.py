"""Utility to convert generic tabular data to a ``TensorDataset``."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def load_tabular_dataset(
    data: Union[str, Path, np.ndarray, pd.DataFrame],
    outcome_col: str = "outcome",
    treatment_col: str = "treatment",
) -> TensorDataset:
    """Convert CSV, NumPy array or ``pandas.DataFrame`` into a ``TensorDataset``.

    Parameters
    ----------
    data:
        Path to a CSV file, a ``numpy.ndarray`` with columns ``X``/``Y``/``T`` or
        a ``pandas.DataFrame`` containing the same information.
    outcome_col:
        Name of the outcome column for ``DataFrame``/CSV inputs.
    treatment_col:
        Name of the treatment column for ``DataFrame``/CSV inputs.

    Returns
    -------
    TensorDataset
        Dataset of ``(X, Y, T)`` tensors.
    """

    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError("NumPy array must have shape (n, d_x + 2)")
        X = data[:, :-2].astype(np.float32)
        Y = data[:, -2].astype(np.float32)
        T = data[:, -1].astype(np.int64)
        return TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(Y).unsqueeze(-1),
            torch.from_numpy(T),
        )
    else:
        raise TypeError("Unsupported data type for load_tabular_dataset")

    if outcome_col not in df.columns or treatment_col not in df.columns:
        raise ValueError("Missing outcome or treatment columns")

    covariate_cols = [c for c in df.columns if c not in {outcome_col, treatment_col}]
    X = df[covariate_cols].to_numpy(dtype=np.float32)
    Y = df[outcome_col].to_numpy(dtype=np.float32)
    T = df[treatment_col].to_numpy(dtype=np.int64)

    return TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(Y).reshape(-1, 1),
        torch.from_numpy(T),
    )


__all__ = ["load_tabular_dataset"]
