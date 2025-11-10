"""
Adaptive benchmark runner using model-specific epoch budgets.

Uses configuration from adaptive_benchmark_config.py to run fair comparisons
where each model gets appropriate training time based on convergence speed.
"""

import json
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from xtylearner.data import get_dataset
from xtylearner.models import get_model, get_model_names
from xtylearner.training import Trainer
from adaptive_benchmark_config import get_epoch_budget, get_sample_size


def run_single_model(model_name: str, dataset_name: str, config: dict) -> dict:
    """Run single model with adaptive epoch budget."""

    print(f"\n{'─'*70}")
    print(f"Model: {model_name} | Dataset: {dataset_name}")

    # Get adaptive parameters
    epochs = get_epoch_budget(model_name)
    n_samples = get_sample_size(dataset_name)

    print(f"Config: {n_samples} samples, {epochs} epochs")
    print(f"{'─'*70}")

    try:
        # Prepare data
        dataset_kwargs = {"n_samples": n_samples, "seed": 42}

        if dataset_name in ["synthetic", "synthetic_mixed"]:
            dataset_kwargs["d_x"] = 2

        if dataset_name == "synthetic_mixed":
            dataset_kwargs["label_ratio"] = 0.5
        elif dataset_name == "criteo_uplift":
            dataset_kwargs["prefer_real"] = False

        full_ds = get_dataset(dataset_name, **dataset_kwargs)

        # Split data
        half = len(full_ds) // 2
        train_ds = TensorDataset(*(t[:half] for t in full_ds.tensors))
        val_ds = TensorDataset(*(t[half:] for t in full_ds.tensors))

        batch_size = config.get("training_config", {}).get("batch_size", 32)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Build model
        x_dim = train_ds.tensors[0].size(1)
        y_dim = train_ds.tensors[1].size(1)

        model_kwargs = {"d_x": x_dim, "d_y": y_dim, "k": 2}
        if model_name == "lp_knn":
            model_kwargs["n_neighbors"] = 3
        elif model_name == "ctm_t":
            model_kwargs = {"d_in": x_dim + y_dim + 1}

        model = get_model(model_name, **model_kwargs)

        # Build optimizer
        if hasattr(model, "loss_G") and hasattr(model, "loss_D"):
            opt = (
                torch.optim.Adam(model.parameters(), lr=0.001),
                torch.optim.Adam(model.parameters(), lr=0.001)
            )
        else:
            params = [p for p in model.parameters() if p.requires_grad]
            if not params:
                params = [torch.zeros(1, requires_grad=True)]
            lr = getattr(model, "default_lr", 0.001)
            opt = torch.optim.Adam(params, lr=lr)

        # Train with adaptive epochs
        start_time = time.time()
        trainer = Trainer(model, opt, train_loader, val_loader=val_loader, logger=None)

        # Report progress every 10 epochs or at milestones
        report_interval = max(1, epochs // 3)
        for epoch_chunk in range(0, epochs, report_interval):
            chunk_size = min(report_interval, epochs - epoch_chunk)
            trainer.fit(chunk_size)

            val_metrics = trainer.evaluate(val_loader)
            print(f"  Epoch {epoch_chunk + chunk_size:2d}/{epochs}: "
                  f"val_rmse={val_metrics['outcome rmse']:.4f}, "
                  f"val_acc={val_metrics['treatment accuracy']:.2f}")

        train_time = time.time() - start_time

        # Final evaluation
        train_metrics = trainer.evaluate(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        result = {
            "model": model_name,
            "dataset": dataset_name,
            "epochs_used": epochs,
            "samples_used": n_samples,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["outcome rmse"],
            "train_acc": train_metrics["treatment accuracy"],
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["outcome rmse"],
            "val_acc": val_metrics["treatment accuracy"],
            "train_time_seconds": train_time,
            "status": "success"
        }

        print(f"  ✓ Final: val_rmse={val_metrics['outcome rmse']:.4f}, time={train_time:.1f}s")

        return result

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {
            "model": model_name,
            "dataset": dataset_name,
            "epochs_used": epochs,
            "samples_used": n_samples,
            "status": "error",
            "error": str(e)
        }


def run_adaptive_benchmark(
    models=None,
    datasets=None,
    output_path="adaptive_benchmark_results.json"
):
    """Run adaptive benchmark on selected models and datasets."""

    # Load config
    config_path = Path("benchmark_config_adaptive.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        print("⚠ Config not found, using defaults")
        config = {"training_config": {"batch_size": 32}}

    # Default to all available models if not specified
    if models is None:
        models = get_model_names()

    # Default datasets
    if datasets is None:
        datasets = ["synthetic", "synthetic_mixed"]

    print("="*80)
    print("ADAPTIVE BENCHMARK")
    print("="*80)
    print(f"\nModels: {len(models)}")
    print(f"Datasets: {datasets}")
    print(f"Using adaptive epoch budgets (5-30 epochs per model)")
    print(f"Sample sizes: 1000 per dataset")
    print()

    results = []
    start_time = time.time()

    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")

        for model_name in models:
            result = run_single_model(model_name, dataset_name, config)
            results.append(result)

    total_time = time.time() - start_time

    # Save results
    output_data = {
        "config": config,
        "metadata": {
            "total_models": len(models),
            "total_datasets": len(datasets),
            "total_time_seconds": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Results saved: {output_path}")

    # Summary by dataset
    for dataset_name in datasets:
        dataset_results = [r for r in results if r.get("dataset") == dataset_name and r.get("status") == "success"]
        if dataset_results:
            print(f"\n{dataset_name.upper()} - Top 10 Models (by val RMSE):")
            print("-" * 70)

            sorted_results = sorted(dataset_results, key=lambda x: x.get("val_rmse", 999))
            for i, result in enumerate(sorted_results[:10], 1):
                print(f"{i:2d}. {result['model']:<25} "
                      f"RMSE: {result['val_rmse']:.4f}  "
                      f"({result['epochs_used']:2d} epochs, {result['train_time_seconds']:.1f}s)")

    return results


if __name__ == "__main__":
    import sys

    # Parse command line args
    if len(sys.argv) > 1:
        # Run specific models
        models = sys.argv[1].split(",")
        print(f"Running benchmark for models: {models}")
    else:
        # Run all models
        models = None
        print("Running benchmark for ALL models")

    results = run_adaptive_benchmark(models=models)
