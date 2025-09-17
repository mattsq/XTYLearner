"""ASV benchmark definitions."""

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


def _all_model_names():
    """Return the list of models exposed to the benchmark suite."""
    names = [model for model in get_model_names() if model != "bridge_diff"]
    if not _HAS_DOUBLEML and "ss_dml" in names:
        names.remove("ss_dml")
    return names


MODEL_NAMES = _all_model_names()


class BenchmarkModels:
    """ASV benchmarks for model validation metrics."""

    timeout = 300  # Increased timeout for complex models

    # Parameters: dataset names and model names
    params = [
        ["synthetic", "synthetic_mixed"],
        MODEL_NAMES,
    ]
    param_names = ["dataset", "model"]
    param_descriptions = {
        "dataset": "Synthetic evaluation datasets shipped with the benchmarks.",
        "model": "Registered model name from ``xtylearner.models``.",
    }

    def track_val_outcome_rmse(self, dataset: str, model: str) -> float:
        """Return validation RMSE for the given model and dataset."""
        row = _run_single((dataset, model))
        if row is None:
            return float("nan")
        return row.get("val outcome rmse", float("nan"))
