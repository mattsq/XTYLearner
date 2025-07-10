"""Utilities for the Twins dataset consisting of US twin births.

The dataset was introduced in the "Inferring Individual Treatment
Effects from Randomized Trials" paper and is often used for evaluating
causal inference methods.  The loader downloads a processed CSV file if
it is not already available.  It returns a ``TensorDataset`` with
covariates ``X`` of shape ``(N, d_x)``, outcomes ``Y`` of shape
``(N, 1)`` and binary treatment indicators ``T`` of shape ``(N,)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset
from urllib.request import urlretrieve

URL = "https://raw.githubusercontent.com/py-why/benchmark-datasets/main/twins/twin_pairs.csv"


def load_twins(data_dir: str = "~/.xtylearner/data") -> TensorDataset:
    """Load the Twins dataset, downloading a processed CSV if needed.

    Parameters
    ----------
    data_dir:
        Location used to cache the dataset file.

    Returns
    -------
    TensorDataset
        Dataset ``(X, Y, T)`` with shapes ``(N, d_x)``, ``(N, 1)`` and ``(N,)``.
    """

    path = Path(data_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / "twin_pairs.csv"
    if not file_path.exists():
        urlretrieve(URL, file_path.as_posix())

    data = np.genfromtxt(file_path, delimiter=",", names=True)
    T = data["treatment"].astype(np.int64)
    Y = data["outcome"].astype(np.float32)
    cov_names = [n for n in data.dtype.names if n not in {"treatment", "outcome"}]
    X = np.vstack([data[n] for n in cov_names]).T.astype(np.float32)

    return TensorDataset(
        torch.from_numpy(X), torch.from_numpy(Y).unsqueeze(-1), torch.from_numpy(T)
    )


__all__ = ["load_twins"]
