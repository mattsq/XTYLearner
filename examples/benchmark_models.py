"""Benchmark all registered models on built-in synthetic datasets."""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ProcessPoolExecutor

from xtylearner.data import get_dataset
from xtylearner.models import get_model, get_model_names
from xtylearner.models.ss_dml import _HAS_DOUBLEML
from xtylearner.training import Trainer


def _make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    params = [p for p in getattr(model, "parameters", lambda: [])() if p.requires_grad]
    if not params:
        params = [torch.zeros(1, requires_grad=True)]
    return torch.optim.Adam(params, lr=0.01)


def _make_gan_optimizer(model: torch.nn.Module):
    opt_g = torch.optim.Adam(model.parameters(), lr=0.01)
    opt_d = torch.optim.Adam(model.parameters(), lr=0.01)
    return opt_g, opt_d


def _run_single(task):
    """Train one model on one dataset and return metrics."""
    ds_name, model_name = task
    if model_name == "ss_dml" and not _HAS_DOUBLEML:
        return None
    torch.set_num_threads(1)
    if ds_name == "synthetic_mixed":
        full_ds = get_dataset(ds_name, n_samples=100, d_x=2, label_ratio=0.5, seed=0)
    else:
        full_ds = get_dataset(ds_name, n_samples=100, d_x=2, seed=0)
    half = len(full_ds) // 2
    ds = TensorDataset(*(t[:half] for t in full_ds.tensors))
    val_ds = TensorDataset(*(t[half:] for t in full_ds.tensors))
    loader = DataLoader(ds, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=10)
    x_dim = ds.tensors[0].size(1)
    y_dim = ds.tensors[1].size(1)
    kwargs = {"d_x": x_dim, "d_y": y_dim, "k": 2}
    if model_name == "lp_knn":
        kwargs["n_neighbors"] = 3
    if model_name == "ctm_t":
        kwargs = {"d_in": x_dim + y_dim + 1}
    model = get_model(model_name, **kwargs)
    if hasattr(model, "loss_G") and hasattr(model, "loss_D"):
        opt = _make_gan_optimizer(model)
    else:
        opt = _make_optimizer(model)
    trainer = Trainer(model, opt, loader, val_loader=val_loader, logger=None)
    trainer.fit(10)
    metrics = trainer.evaluate(loader)
    val_metrics = trainer.evaluate(val_loader)
    row = {"dataset": ds_name, "model": model_name}
    row.update({f"train {k}": v for k, v in metrics.items()})
    row.update({f"val {k}": v for k, v in val_metrics.items()})
    return row


def run_benchmark(output_path: str = "benchmark_results.md") -> None:
    """Train every model for ten epochs and record metrics."""
    dataset_names = ["synthetic", "synthetic_mixed"]
    tasks = [
        (ds_name, model_name)
        for ds_name in dataset_names
        for model_name in get_model_names()
    ]
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        for row in pool.map(_run_single, tasks):
            if row is not None:
                results.append(row)
    df = pd.DataFrame(results)
    if "val outcome rmse" in df.columns:
        df = df.sort_values("val outcome rmse")
    elif "train outcome rmse" in df.columns:
        df = df.sort_values("train outcome rmse")
    df.to_markdown(output_path, index=False)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    run_benchmark()
