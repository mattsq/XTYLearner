"""Utility to convert generic tabular data to a ``TensorDataset``."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def load_tabular_dataset(
    data: Union[str, Path, np.ndarray, pd.DataFrame],
    outcome_col: Union[str, Sequence[str], int] = "outcome",
    treatment_col: str = "treatment",
) -> TensorDataset:
    """Convert CSV, NumPy array or ``pandas.DataFrame`` into a ``TensorDataset``.

    Parameters
    ----------
    data:
        Path to a CSV file, a ``numpy.ndarray`` with columns ``X``/``Y``/``T`` or
        a ``pandas.DataFrame`` containing the same information.
    outcome_col:
        Name of the outcome column for ``DataFrame``/CSV inputs.  Can be a
        sequence of names if multiple outcome variables are present.  When
        ``data`` is a ``numpy.ndarray`` this argument may also be an integer
        specifying the number of outcome columns.  In this case the outcome
        columns are assumed to appear immediately before the treatment column.
    treatment_col:
        Name of the treatment column for ``DataFrame``/CSV inputs.

    Returns
    -------
    TensorDataset
        Dataset of ``(X, Y, T)`` tensors. Columns not designated as outcome or
        treatment become covariates ``X`` in the original order. When ``T``
        contains string categories, a ``treatment_mapping`` attribute on the
        returned dataset stores the category-to-index mapping.
    """

    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, np.ndarray):
        if isinstance(outcome_col, str):
            n_outcomes = 1
        elif isinstance(outcome_col, int):
            n_outcomes = outcome_col
        else:
            n_outcomes = len(outcome_col)

        if data.ndim != 2 or data.shape[1] < n_outcomes + 2:
            raise ValueError("NumPy array must have shape (n, d_x + n_outcomes + 1)")

        X = data[:, : -(n_outcomes + 1)].astype(np.float32)
        Y = data[:, -(n_outcomes + 1) : -1].astype(np.float32)
        T = data[:, -1].astype(np.int64)
        return TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(Y).reshape(-1, n_outcomes),
            torch.from_numpy(T),
        )
    else:
        raise TypeError("Unsupported data type for load_tabular_dataset")

    outcome_cols: Sequence[str]
    if isinstance(outcome_col, str):
        outcome_cols = [outcome_col]
    elif isinstance(outcome_col, int):  # type: ignore[unreachable]
        raise TypeError("Numeric outcome_col only valid for NumPy array inputs")
    else:
        outcome_cols = list(outcome_col)

    missing = set(outcome_cols + [treatment_col]) - set(df.columns)
    if missing:
        raise ValueError("Missing outcome or treatment columns")

    covariate_cols = [
        c for c in df.columns if c not in set(outcome_cols + [treatment_col])
    ]
    X = df[covariate_cols].to_numpy(dtype=np.float32)
    Y = df[outcome_cols].to_numpy(dtype=np.float32)

    treatment_series = df[treatment_col]
    treatment_mapping = None
    if not np.issubdtype(treatment_series.dtype, np.number):
        categories = sorted(treatment_series.unique())
        treatment_mapping = {c: i for i, c in enumerate(categories)}
        T = treatment_series.map(treatment_mapping).to_numpy(dtype=np.int64)
    else:
        T = treatment_series.to_numpy(dtype=np.int64)

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(Y).reshape(-1, len(outcome_cols)),
        torch.from_numpy(T),
    )

    if treatment_mapping is not None:
        dataset.treatment_mapping = treatment_mapping

    return dataset


__all__ = ["load_tabular_dataset"]
