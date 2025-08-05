"""Utility functions for the IHDP (Infant Health and Development Program) dataset.

The loader downloads the semi-synthetic dataset released with the
``CEVAE`` paper if it is not already present locally.  It returns a
``TensorDataset`` containing covariates ``X``, outcomes ``Y`` and
treatment labels ``T`` with shapes ``(N, d_x)``, ``(N, 1)`` and
``(N,)`` respectively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import TensorDataset
from urllib.request import urlretrieve

# Original ``.npz`` files hosted with the CEVAE repository were removed from the
# default branch.  The project now provides per-replicate CSV files instead.
# We load the first two replicates for the training and test splits
# respectively.
URLS = {
    "train": "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv",
    "test": "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_2.csv",
}


def load_ihdp(
    split: Literal["train", "test"] = "train",
    data_dir: str = "~/.xtylearner/data",
    *,
    continuous_treatment: bool = False,
) -> TensorDataset:
    """Load the IHDP dataset provided with the CEVAE benchmark.

    Parameters
    ----------
    split:
        Which portion of the dataset to load, ``"train"`` or ``"test"``.
    data_dir:
        Directory where the CSV files are stored or should be downloaded to.
    continuous_treatment:
        If ``True`` keep the treatment array as ``np.float32`` and return a
        floating-point tensor. By default integer labels are returned.

    Returns
    -------
    TensorDataset
        TensorDataset with covariates ``X``, outcomes ``Y`` and treatment ``T``.
        When ``continuous_treatment=True`` the treatment tensor has dtype
        ``torch.float32``.
    """

    url = URLS[split]
    path = Path(data_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / Path(url).name
    if not file_path.exists():
        urlretrieve(url, file_path.as_posix())

    data = np.loadtxt(file_path, delimiter=",")
    # Columns: t, y_factual, y_cfactual, mu0, mu1, x1..x25
    X = torch.from_numpy(data[:, 5:]).float()
    Y = torch.from_numpy(data[:, 1]).float().unsqueeze(-1)
    t_np = data[:, 0]
    if continuous_treatment:
        t_np = t_np.astype(np.float32)
        T = torch.from_numpy(t_np).float()
    else:
        t_np = t_np.astype(np.int64)
        T = torch.from_numpy(t_np).long()

    return TensorDataset(X, Y, T)


__all__ = ["load_ihdp"]
