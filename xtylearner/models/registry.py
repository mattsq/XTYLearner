"""Simple registry for constructing model classes by name."""

from typing import Callable, Dict, Any, List
import inspect

# Mapping from model names to their constructors.  Model files register
# themselves here via the :func:`register_model` decorator below.
_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a model class under ``name``."""

    def decorator(cls: Callable[..., Any]) -> Callable[..., Any]:
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **hparams: Any) -> Any:
    """Instantiate a model from the registry."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**hparams)


def get_model_names() -> list[str]:
    """Return a sorted list of registered model names."""

    return sorted(_MODEL_REGISTRY)


def get_model_args(name: str) -> List[str]:
    """Return the constructor argument names for a registered model."""

    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}"
        )
    cls = _MODEL_REGISTRY[name]
    sig = inspect.signature(cls)
    return [p.name for p in sig.parameters.values() if p.name != "self"]


__all__ = ["register_model", "get_model", "get_model_names", "get_model_args"]
