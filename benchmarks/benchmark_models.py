"""ASV benchmark definitions."""

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


def get_benchmark_models():
    """Get models to benchmark based on environment variable or default to cycle_dual only."""
    # Check environment variable for model list
    env_models = os.environ.get("BENCHMARK_MODELS", "").strip()
    if env_models:
        # Split comma-separated model list and clean up whitespace
        models = [model.strip() for model in env_models.split(",") if model.strip()]
        if models:
            return models
    
    # For debugging: only run cycle_dual model if no env var specified
    return ["cycle_dual"]


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
