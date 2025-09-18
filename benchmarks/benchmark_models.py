"""ASV benchmark definitions."""

import math
import os
from pathlib import Path
import sys

# Ensure repository root on ``sys.path`` so ``examples`` can be imported when
# ASV discovers benchmarks.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmark_models import _run_single  # noqa: E402
from xtylearner.models import get_model_names  # noqa: E402
from xtylearner.models.ss_dml import _HAS_DOUBLEML  # noqa: E402


# Core models to always benchmark for regression detection
CORE_MODELS = ["jsbf", "eg_ddi", "cevae_m"]


def _parse_int_env(var_name: str, default: int) -> int:
    """Return an integer environment variable or a default value."""

    value = os.environ.get(var_name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _apply_chunking(models: list[str]) -> list[str]:
    """Return the subset of models assigned to the current chunk."""

    if not models:
        return models

    chunk_total = max(1, _parse_int_env("MODEL_CHUNK_TOTAL", 1))
    chunk_index = _parse_int_env("MODEL_CHUNK_INDEX", 0)

    if chunk_total <= 1 or len(models) <= 1:
        return models

    # Avoid requesting more chunks than there are models. When the matrix in the
    # workflow over-provisions chunks we simply skip the surplus ones by
    # returning an empty list.
    chunk_total = min(chunk_total, len(models))
    if chunk_index < 0 or chunk_index >= chunk_total:
        return []

    chunk_size = math.ceil(len(models) / chunk_total)
    start = chunk_size * chunk_index
    end = start + chunk_size
    return models[start:end]


def get_benchmark_models():
    """Get models to benchmark based on environment variable or default to cycle_dual only."""
    # Check environment variable for model list
    env_models = os.environ.get("BENCHMARK_MODELS", "").strip()
    if env_models:
        # Split comma-separated model list and clean up whitespace
        models = [model.strip() for model in env_models.split(",") if model.strip()]
        if models:
            return _apply_chunking(models)

    # For debugging: only run cycle_dual model if no env var specified
    return _apply_chunking(["cycle_dual"])


MODEL_NAMES = get_benchmark_models()


class BenchmarkModels:
    """ASV benchmarks for model validation metrics."""

    timeout = 600  # 10 minutes for complex ML models

    # Parameters: dataset names and model names
    params = [
        ["synthetic", "synthetic_mixed"],
        MODEL_NAMES,
    ]
    param_names = ["dataset", "model"]

    def track_val_outcome_rmse(self, dataset: str, model: str) -> float:
        """Return validation RMSE for the given model and dataset."""
        row = _run_single((dataset, model))
        if row is None:
            return float("nan")
        return row.get("val outcome rmse", float("nan"))
