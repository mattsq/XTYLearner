"""Registry for active learning query strategies used in benchmarking."""

from __future__ import annotations

from typing import Callable, Dict, Type

import torch

from .strategies import (
    QueryStrategy,
    EntropyT,
    DeltaCATE,
    CATEUncertainty,
    ConformalCATEIntervalStrategy,
    DebiasedCoverageAcquisition,
    FCCMRadius,
)


class RandomStrategy(QueryStrategy):
    """Return uniform random acquisition scores for the pool."""

    def forward(
        self,
        model,  # type: ignore[override]
        X_unlab: torch.Tensor,
        rep_fn,
        batch_size: int,
    ) -> torch.Tensor:
        del model, rep_fn, batch_size
        device = X_unlab.device
        return torch.rand(len(X_unlab), device=device)


_STRATEGY_ALIASES: Dict[str, Type[QueryStrategy] | Callable[..., QueryStrategy]] = {
    "entropy_t": EntropyT,
    "delta_cate": DeltaCATE,
    "cate_uncertainty": CATEUncertainty,
    "conformal_cate_interval": ConformalCATEIntervalStrategy,
    "debiased_coverage": DebiasedCoverageAcquisition,
    "fccm_radius": FCCMRadius,
}


def get_strategy(name: str, **kwargs) -> QueryStrategy:
    """Return an instantiated active learning strategy by name.

    Parameters
    ----------
    name:
        Strategy identifier.  Supports registered strategy class names and the
        special baselines ``"random"`` and ``"passive_iid"`` (not yet
        implemented).
    **kwargs:
        Keyword arguments forwarded to the strategy constructor.

    Returns
    -------
    QueryStrategy
        Instantiated strategy ready for use with :class:`ActiveTrainer`.
    """

    key = name.lower()
    if key == "random":
        return RandomStrategy()

    if key == "passive_iid":
        raise NotImplementedError(
            "The 'passive_iid' baseline is not implemented yet."
        )

    if key in _STRATEGY_ALIASES:
        strategy_ctor = _STRATEGY_ALIASES[key]
        return strategy_ctor(**kwargs)

    # Allow direct access via the STRATEGIES mapping exposed in strategies.py
    from .strategies import STRATEGIES

    if key in STRATEGIES:
        return STRATEGIES[key](**kwargs)

    raise ValueError(f"Unknown active learning strategy '{name}'")


__all__ = ["get_strategy", "RandomStrategy"]

