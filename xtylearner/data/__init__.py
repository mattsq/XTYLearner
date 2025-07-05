"""Dataset utilities for XTYLearner."""

from __future__ import annotations

from typing import Callable, Dict

from .toy_dataset import load_toy_dataset
from .synthetic_dataset import load_synthetic_dataset
from .mixed_synthetic_dataset import load_mixed_synthetic_dataset
from .ihdp_dataset import load_ihdp
from .twins_dataset import load_twins
from .tabular_dataset import load_tabular_dataset


_DATASETS: Dict[str, Callable[..., object]] = {
    "toy": load_toy_dataset,
    "synthetic": load_synthetic_dataset,
    "synthetic_mixed": load_mixed_synthetic_dataset,
    "ihdp": load_ihdp,
    "twins": load_twins,
}


def get_dataset(name: str, **kwargs):
    """Load one of the built-in datasets by name.

    Parameters
    ----------
    name:
        Name of the dataset.  One of ``"toy"``, ``"synthetic"``,
        ``"ihdp"`` or ``"twins"``.
    **kwargs:
        Additional keyword arguments forwarded to the dataset loader.

    Returns
    -------
    ``torch.utils.data.Dataset``
        The requested dataset object.
    """

    name = name.lower()
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset '{name}'")
    return _DATASETS[name](**kwargs)


__all__ = [
    "get_dataset",
    "load_toy_dataset",
    "load_synthetic_dataset",
    "load_mixed_synthetic_dataset",
    "load_ihdp",
    "load_twins",
    "load_tabular_dataset",
]
