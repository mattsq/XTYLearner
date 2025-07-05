"""Configuration files for experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the ``.yaml`` file to load.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """

    with Path(path).expanduser().open("r") as f:
        return yaml.safe_load(f)


__all__ = ["load"]
