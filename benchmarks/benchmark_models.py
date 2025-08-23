from examples.benchmark_models import _run_single
from xtylearner.models import get_model_names


class BenchmarkModels:
    """ASV benchmarks for model validation metrics."""

    # Parameters: dataset names and model names
    params = [
        ["synthetic", "synthetic_mixed"],
        get_model_names(),
    ]
    param_names = ["dataset", "model"]

    def track_val_outcome_rmse(self, dataset: str, model: str) -> float:
        """Return validation RMSE for the given model and dataset."""
        row = _run_single((dataset, model))
        return row.get("val outcome rmse", float("nan"))
