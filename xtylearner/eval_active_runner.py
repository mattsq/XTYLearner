"""Utilities for running active learning benchmarking experiments."""

from __future__ import annotations

import importlib
import math
import random
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .active.registry import get_strategy
from .active.utils import predict_outcome
from .data.utils_active_split import make_active_splits
from .models import get_model
from .training import ActiveTrainer


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_dataset_loader(name: str):
    module = importlib.import_module("xtylearner.data")
    if not hasattr(module, name):
        raise ValueError(f"Unknown dataset loader '{name}'")
    return getattr(module, name)


def _load_active_dataset(
    dataset_conf: Dict[str, Any],
    seed: int,
) -> tuple[TensorDataset, TensorDataset, Dict[str, Any]]:
    loader_name = dataset_conf["loader"]
    loader = _resolve_dataset_loader(loader_name)
    dataset_args = dict(dataset_conf.get("args", {}))
    dataset_args.setdefault("seed", seed)
    data_obj = loader(**dataset_args)

    metadata: Dict[str, Any] = {}
    if isinstance(data_obj, (list, tuple)) and len(data_obj) >= 2:
        pool_dataset = data_obj[0]
        test_dataset = data_obj[1]
        if not isinstance(pool_dataset, TensorDataset) or not isinstance(
            test_dataset, TensorDataset
        ):
            raise TypeError(
                "Dataset loader must return TensorDataset instances for active benchmarking"
            )
        if len(data_obj) >= 3 and isinstance(data_obj[2], dict):
            metadata = dict(data_obj[2])
        return pool_dataset, test_dataset, metadata

    if isinstance(data_obj, TensorDataset):
        test_fraction = dataset_conf.get("test_fraction", 0.2)
        return make_active_splits(data_obj, test_fraction=test_fraction, seed=seed)

    raise TypeError(
        "Active dataset loaders must return a TensorDataset or a tuple of TensorDatasets"
    )


def _build_optimizer(model: torch.nn.Module, lr: float = 1e-3):
    if hasattr(model, "loss_G") and hasattr(model, "loss_D"):
        opt_g = torch.optim.Adam(model.parameters(), lr=lr)
        opt_d = torch.optim.Adam(model.parameters(), lr=lr)
        return opt_g, opt_d

    params = [p for p in getattr(model, "parameters", lambda: [])() if p.requires_grad]
    if not params:
        params = [torch.zeros(1, requires_grad=True)]
    return torch.optim.Adam(params, lr=lr)


def _extract_metadata_tensor(
    metadata: Dict[str, Any],
    keys: Sequence[str],
    expected_length: int,
) -> torch.Tensor | None:
    if not metadata:
        return None
    value = None
    for key in keys:
        if key in metadata:
            value = metadata[key]
            break
    if value is None:
        return None
    tensor = torch.as_tensor(value)
    indices = metadata.get("test_indices")
    if tensor.numel() != expected_length and indices is not None:
        idx_tensor = torch.as_tensor(indices)
        if tensor.dim() == 1:
            if tensor.numel() > idx_tensor.max().item():
                tensor = tensor[idx_tensor]
        elif tensor.dim() >= 2:
            if tensor.size(0) > idx_tensor.max().item():
                tensor = tensor[idx_tensor]
    if tensor.numel() != expected_length:
        return None
    return tensor


def evaluate_model_on_test(
    trainer: ActiveTrainer,
    test_dataset: TensorDataset,
    metadata: Dict[str, Any] | None = None,
    *,
    batch_size: int = 256,
) -> Dict[str, float]:
    loader = DataLoader(test_dataset, batch_size=max(1, batch_size))
    base_metrics = trainer.evaluate(loader)

    metrics: Dict[str, float] = {}
    metrics["loss"] = float(base_metrics.get("loss", math.nan))
    metrics["Y_RMSE"] = float(base_metrics.get("outcome rmse", math.nan))
    metrics["T_acc"] = float(base_metrics.get("treatment accuracy", math.nan))

    device = getattr(trainer._trainer, "device", "cpu")  # type: ignore[attr-defined]
    model = trainer._trainer.model  # type: ignore[attr-defined]

    tau_true = None
    y0_true = None
    y1_true = None
    if metadata:
        tau_true = _extract_metadata_tensor(metadata, ("true_tau", "tau", "cate"), len(test_dataset))
        y0_true = _extract_metadata_tensor(metadata, ("y0", "mu0"), len(test_dataset))
        y1_true = _extract_metadata_tensor(metadata, ("y1", "mu1"), len(test_dataset))

    has_binary = False
    if len(test_dataset) > 0:
        t_tensor = test_dataset.tensors[2]
        unique_vals = torch.unique(t_tensor.cpu())
        if all(val in (0, 1) for val in unique_vals.tolist()):
            has_binary = True

    preds_tau: List[torch.Tensor] = []
    preds_y0: List[torch.Tensor] = []
    preds_y1: List[torch.Tensor] = []
    obs_y0: List[torch.Tensor] = []
    obs_y1: List[torch.Tensor] = []

    if has_binary:
        for batch in loader:
            X, Y, T = [b.to(device) for b in batch]
            with torch.no_grad():
                t0 = torch.zeros(len(X), dtype=torch.long, device=device)
                t1 = torch.ones(len(X), dtype=torch.long, device=device)
                y_hat0 = predict_outcome(model, X, t0)
                y_hat1 = predict_outcome(model, X, t1)
            preds_y0.append(y_hat0.detach().cpu().view(len(X), -1))
            preds_y1.append(y_hat1.detach().cpu().view(len(X), -1))
            preds_tau.append((y_hat1 - y_hat0).detach().cpu().view(len(X), -1))

            if torch.is_floating_point(T):
                mask0 = torch.isclose(T, torch.zeros_like(T))
                mask1 = torch.isclose(T, torch.ones_like(T))
            else:
                mask0 = T == 0
                mask1 = T == 1
            if mask0.any():
                obs_y0.append(Y[mask0].detach().cpu().view(-1))
            if mask1.any():
                obs_y1.append(Y[mask1].detach().cpu().view(-1))

    if preds_y0 and preds_y1:
        y0_cat = torch.cat(preds_y0).view(-1)
        y1_cat = torch.cat(preds_y1).view(-1)
        ate_pred = y1_cat.mean() - y0_cat.mean()
        if obs_y0 and obs_y1:
            y0_obs = torch.cat(obs_y0).view(-1)
            y1_obs = torch.cat(obs_y1).view(-1)
            ate_true = y1_obs.mean() - y0_obs.mean()
            metrics["ATE_abs_err"] = float(torch.abs(ate_pred - ate_true).item())
        else:
            metrics["ATE_abs_err"] = float("nan")

        if tau_true is not None:
            tau_true = tau_true.view(-1).float()
            tau_pred = (y1_cat - y0_cat).view(-1)
            if len(tau_true) == len(tau_pred):
                diff = tau_pred - tau_true
                metrics["PEHE"] = float(torch.sqrt((diff ** 2).mean()).item())
            else:
                metrics["PEHE"] = float("nan")
        elif y0_true is not None and y1_true is not None:
            tau_true = (y1_true - y0_true).view(-1).float()
            tau_pred = (y1_cat - y0_cat).view(-1)
            if len(tau_true) == len(tau_pred):
                diff = tau_pred - tau_true
                metrics["PEHE"] = float(torch.sqrt((diff ** 2).mean()).item())
            else:
                metrics["PEHE"] = float("nan")
        else:
            metrics.setdefault("PEHE", float("nan"))
    else:
        metrics.setdefault("ATE_abs_err", float("nan"))
        metrics.setdefault("PEHE", float("nan"))

    return metrics


def run_active_benchmark_once(
    dataset_conf: Dict[str, Any],
    model_conf: Dict[str, Any],
    strategy_conf: Dict[str, Any],
    budget: int,
    batch: int,
    epochs_per_round: int,
    seed: int,
) -> Dict[str, Any]:
    _set_random_seed(seed)

    pool_dataset, test_dataset, metadata = _load_active_dataset(dataset_conf, seed)

    batch_size = dataset_conf.get("batch_size", 64)
    train_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=True)

    model = get_model(model_conf["name"], **model_conf.get("args", {}))
    optimizer = _build_optimizer(model)
    strategy = get_strategy(strategy_conf["name"], **strategy_conf.get("args", {}))

    trainer = ActiveTrainer(
        model,
        optimizer,
        train_loader,
        strategy=strategy,
        budget=budget,
        batch=batch,
    )

    round_logs: List[Dict[str, Any]] = []
    for state in trainer.iterate_rounds(epochs_per_round):
        metrics = evaluate_model_on_test(trainer, test_dataset, metadata)
        round_logs.append(
            {
                "round": state["round"],
                "labels_used": state["labels_used"],
                "metrics": metrics,
            }
        )

    final_metrics = evaluate_model_on_test(trainer, test_dataset, metadata)

    return {
        "round_metrics": round_logs,
        "final_metrics": final_metrics,
        "seed": seed,
    }


def _nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _nanstd(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return float("nan")
    return float(arr.std(ddof=0))


def aggregate_across_seeds(runs_for_strategy: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs_for_strategy:
        return {
            "round_metrics_mean": [],
            "round_metrics_std": [],
            "AULC": {},
            "final_metrics_mean": {},
            "final_metrics_std": {},
        }

    round_ids = sorted(
        {
            entry["round"]
            for run in runs_for_strategy
            for entry in run.get("round_metrics", [])
        }
    )
    metric_names = sorted(
        {
            name
            for run in runs_for_strategy
            for entry in run.get("round_metrics", [])
            for name in entry.get("metrics", {}).keys()
        }
    )

    round_metrics_mean = []
    round_metrics_std = []
    for round_id in round_ids:
        label_counts: List[float] = []
        metric_values: Dict[str, List[float]] = {name: [] for name in metric_names}
        for run in runs_for_strategy:
            matched = next(
                (entry for entry in run.get("round_metrics", []) if entry["round"] == round_id),
                None,
            )
            if matched is None:
                continue
            label_counts.append(float(matched.get("labels_used", math.nan)))
            for name in metric_names:
                metric_values[name].append(matched.get("metrics", {}).get(name, math.nan))

        mean_entry = {
            "round": round_id,
            "labels_used": _nanmean(label_counts),
            "metrics": {name: _nanmean(vals) for name, vals in metric_values.items()},
        }
        std_entry = {
            "round": round_id,
            "labels_used": _nanstd(label_counts),
            "metrics": {name: _nanstd(vals) for name, vals in metric_values.items()},
        }
        round_metrics_mean.append(mean_entry)
        round_metrics_std.append(std_entry)

    aulc: Dict[str, float] = {}
    for name in metric_names:
        per_seed_means: List[float] = []
        for run in runs_for_strategy:
            vals = [entry.get("metrics", {}).get(name, math.nan) for entry in run.get("round_metrics", [])]
            arr = np.asarray(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size:
                per_seed_means.append(float(arr.mean()))
        aulc[name] = float(np.mean(per_seed_means)) if per_seed_means else float("nan")

    final_metric_names = sorted(
        {
            name for run in runs_for_strategy for name in run.get("final_metrics", {}).keys()
        }
    )
    final_mean = {}
    final_std = {}
    for name in final_metric_names:
        vals = [run.get("final_metrics", {}).get(name, math.nan) for run in runs_for_strategy]
        final_mean[name] = _nanmean(vals)
        final_std[name] = _nanstd(vals)

    return {
        "round_metrics_mean": round_metrics_mean,
        "round_metrics_std": round_metrics_std,
        "AULC": aulc,
        "final_metrics_mean": final_mean,
        "final_metrics_std": final_std,
    }


__all__ = [
    "run_active_benchmark_once",
    "aggregate_across_seeds",
    "evaluate_model_on_test",
]

