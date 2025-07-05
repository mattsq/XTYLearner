"""Simple registry for constructing model classes by name."""

from typing import Callable, Dict, Any

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
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[name](**hparams)


__all__ = ["register_model", "get_model"]
