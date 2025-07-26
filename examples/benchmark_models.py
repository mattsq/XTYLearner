"""Benchmark all registered models on built-in synthetic datasets."""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from xtylearner.data import get_dataset
from xtylearner.models import get_model, get_model_names
from xtylearner.models.ss_dml import _HAS_DOUBLEML
from xtylearner.training import Trainer, ConsoleLogger


def _make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    params = [p for p in getattr(model, "parameters", lambda: [])() if p.requires_grad]
    if not params:
        params = [torch.zeros(1, requires_grad=True)]
    return torch.optim.Adam(params, lr=0.01)


def _make_gan_optimizer(model: torch.nn.Module):
    opt_g = torch.optim.Adam(model.parameters(), lr=0.01)
    opt_d = torch.optim.Adam(model.parameters(), lr=0.01)
    return opt_g, opt_d


def run_benchmark(output_path: str = "benchmark_results.md") -> None:
    """Train every model for ten epochs and record metrics."""
    dataset_names = ["synthetic", "synthetic_mixed"]
    results = []
    for ds_name in dataset_names:
        if ds_name == "synthetic_mixed":
            ds = get_dataset(ds_name, n_samples=50, d_x=2, label_ratio=0.5)
        else:
            ds = get_dataset(ds_name, n_samples=50, d_x=2)
        loader = DataLoader(ds, batch_size=10, shuffle=True)
        x_dim = ds.tensors[0].size(1)
        y_dim = ds.tensors[1].size(1)
        for model_name in get_model_names():
            if model_name == "ss_dml" and not _HAS_DOUBLEML:
                continue
            print(f"Running {model_name} on {ds_name}")
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
            logger = ConsoleLogger(print_every=1)
            trainer = Trainer(model, opt, loader, logger=logger)
            trainer.fit(10)
            metrics = trainer.evaluate(loader)
            results.append({"dataset": ds_name, "model": model_name, **metrics})
    df = pd.DataFrame(results)
    if "outcome rmse" in df.columns:
        df = df.sort_values("outcome rmse")
    df.to_markdown(output_path, index=False)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    run_benchmark()
