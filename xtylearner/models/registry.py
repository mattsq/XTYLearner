"""Simple registry for constructing model classes by name."""

from typing import Type, Dict

from .cycle_dual import CycleDual
from .flow_ssc import MixtureOfFlows
from .multitask_selftrain import MultiTask

_MODEL_REGISTRY: Dict[str, Type] = {
    "cycle_dual": CycleDual,
    "flow_ssc": MixtureOfFlows,
    "multitask": MultiTask,
}


def get_model(name: str, **hparams):
    """Instantiate a model from the registry."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**hparams)

__all__ = ["get_model"]
