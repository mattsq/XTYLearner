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

URLS = {
    "train": "https://github.com/AMLab-Amsterdam/CEVAE/raw/master/datasets/ihdp_npci_1-100.train.npz",
    "test": "https://github.com/AMLab-Amsterdam/CEVAE/raw/master/datasets/ihdp_npci_1-100.test.npz",
}


def load_ihdp(
    split: Literal["train", "test"] = "train",
    data_dir: str = "~/.xtylearner/data",
) -> TensorDataset:
    """Load the IHDP dataset provided with the CEVAE benchmark.

    Parameters
    ----------
    split:
        Which portion of the dataset to load, ``"train"`` or ``"test"``.
    data_dir:
        Directory where the ``.npz`` files are stored or should be downloaded to.

    Returns
    -------
    TensorDataset
        TensorDataset with covariates ``X``, outcomes ``Y`` and treatment ``T``.
    """

    url = URLS[split]
    path = Path(data_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / Path(url).name
    if not file_path.exists():
        urlretrieve(url, file_path.as_posix())

    data = np.load(file_path)
    X = torch.from_numpy(data["x"]).float()
    Y = torch.from_numpy(data["yf"]).float().unsqueeze(-1)
    T = torch.from_numpy(data["t"]).long()

    return TensorDataset(X, Y, T)


__all__ = ["load_ihdp"]
