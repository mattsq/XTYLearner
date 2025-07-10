"""Simple registry for constructing model classes by name."""

from typing import Callable, Dict, Any, List
import inspect

# Mapping from model names to their constructors.  Model files register
# themselves here via the :func:`register_model` decorator below.
_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register ``cls`` under ``name`` so it can be constructed later.

    Parameters
    ----------
    name:
        Identifier used with :func:`get_model`.

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        A decorator that adds the class to the registry unchanged.
    """

    def decorator(cls: Callable[..., Any]) -> Callable[..., Any]:
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **hparams: Any) -> Any:
    """Instantiate a registered model.

    Parameters
    ----------
    name:
        Model identifier previously passed to :func:`register_model`.
    **hparams:
        Keyword arguments forwarded to the model constructor.

    Returns
    -------
    Any
        A newly created model instance.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**hparams)


def get_model_names() -> list[str]:
    """List names of all registered models in alphabetical order."""

    return sorted(_MODEL_REGISTRY)


def get_model_args(name: str) -> List[str]:
    """Return the argument names for ``name``'s constructor.

    Parameters
    ----------
    name:
        Identifier of a registered model.

    Returns
    -------
    list[str]
        Positional parameter names excluding ``self``.
    """

    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}"
        )
    cls = _MODEL_REGISTRY[name]
    sig = inspect.signature(cls)
    return [p.name for p in sig.parameters.values() if p.name != "self"]


__all__ = ["register_model", "get_model", "get_model_names", "get_model_args"]
