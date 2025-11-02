"""Helpers for preparing active learning dataset splits."""

from __future__ import annotations

from typing import Tuple, Dict, Any

import torch
from torch.utils.data import TensorDataset


def make_active_splits(
    dataset: TensorDataset,
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[TensorDataset, TensorDataset, Dict[str, Any]]:
    """Split ``dataset`` into pool/test subsets for active learning.

    Parameters
    ----------
    dataset:
        ``TensorDataset`` containing ``(X, Y, T)`` where missing treatments are
        encoded as ``-1``.  Additional tensors are ignored.
    test_fraction:
        Fraction of labelled samples to allocate to the held-out test set.
    seed:
        Random seed controlling the split.

    Returns
    -------
    tuple
        ``(pool_dataset, test_dataset, metadata)`` where metadata contains the
        ``test_indices`` used for splitting.
    """

    if not isinstance(dataset, TensorDataset):
        raise TypeError("make_active_splits expects a TensorDataset input")

    tensors = dataset.tensors
    if len(tensors) < 3:
        raise ValueError("Active learning datasets must provide X, Y, T tensors")

    X, Y, T = tensors[:3]
    n = len(X)
    if len(Y) != n or len(T) != n:
        raise ValueError("Inconsistent tensor lengths in dataset")

    labelled_mask: torch.Tensor
    if T.dim() == 1:
        labelled_mask = T >= 0
    else:
        labelled_mask = torch.isfinite(T).all(-1)

    labelled_indices = torch.nonzero(labelled_mask, as_tuple=False).view(-1)
    if len(labelled_indices) == 0:
        labelled_indices = torch.arange(n)

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = labelled_indices[torch.randperm(len(labelled_indices), generator=generator)]
    test_size = max(int(len(labelled_indices) * test_fraction), 1)
    test_size = min(test_size, len(labelled_indices))
    test_indices = perm[:test_size]

    pool_mask = torch.ones(n, dtype=torch.bool)
    pool_mask[test_indices] = False

    pool_tensors = [tensor[pool_mask].clone() for tensor in tensors[:3]]
    test_tensors = [tensor[test_indices].clone() for tensor in tensors[:3]]

    pool_dataset = TensorDataset(*pool_tensors)
    test_dataset = TensorDataset(*test_tensors)

    metadata: Dict[str, Any] = {"test_indices": test_indices.clone()}
    return pool_dataset, test_dataset, metadata


__all__ = ["make_active_splits"]

