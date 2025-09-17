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
    """Get models to benchmark based on environment variable or default to all."""
    # Check for selective benchmarking
    env_models = os.environ.get('BENCHMARK_MODELS', '').strip()
    chunk_index = os.environ.get('MODEL_CHUNK_INDEX', '').strip()
    total_chunks = int(os.environ.get('MODEL_CHUNK_TOTAL', '1'))
    
    # Get base model list
    all_models = [m for m in get_model_names() if m != "bridge_diff"]
    if not _HAS_DOUBLEML and "ss_dml" in all_models:
        all_models.remove("ss_dml")
    
    # Selective benchmarking based on changed files
    if env_models:
        selected_models = [m.strip() for m in env_models.split(',') if m.strip()]
        # Add core models for regression detection
        selected_models.extend(CORE_MODELS)
        # Remove duplicates while preserving order
        return list(dict.fromkeys(m for m in selected_models if m in all_models))
    
    # Parallel execution chunking
    if chunk_index:
        chunk_idx = int(chunk_index)
        chunk_size = len(all_models) // total_chunks
        remainder = len(all_models) % total_chunks
        
        # Calculate start and end indices for this chunk
        start_idx = chunk_idx * chunk_size + min(chunk_idx, remainder)
        end_idx = start_idx + chunk_size + (1 if chunk_idx < remainder else 0)
        
        return all_models[start_idx:end_idx]
    
    # Default: return all models
    return all_models


MODEL_NAMES = get_benchmark_models()


class BenchmarkModels:
    """ASV benchmarks for model validation metrics."""

    timeout = 300  # Increased timeout for complex models

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
