"""
Experimental benchmark: 1000 samples, 30 epochs
Compare against current CI baseline (100 samples, 10 epochs)
"""

import json
import time
import torch
from torch.utils.data import DataLoader, TensorDataset

from xtylearner.data import get_dataset
from xtylearner.models import get_model
from xtylearner.training import Trainer


def run_experiment():
    """Run targeted benchmark with increased samples and epochs."""

    # Selected models for comparison
    models_to_test = [
        # Top performers (fast convergers)
        "prob_circuit",
        "vat",
        "mean_teacher",
        "fixmatch",
        "ganite",
        "cycle_dual",

        # Mid-tier
        "flow_ssc",
        "multitask",

        # Diffusion models (slow convergers)
        "bridge_diff",
        "diffusion_cevae",
        "lt_flow_diff",

        # Other interesting models
        "m2_vae",
        "ccl_cpc",
    ]

    datasets_to_test = ["synthetic", "synthetic_mixed"]

    results = {
        "config": {
            "n_samples": 1000,
            "epochs": 30,
            "batch_size": 32,
            "description": "Experimental benchmark with increased data and training"
        },
        "results": []
    }

    print("="*80)
    print("EXPERIMENTAL BENCHMARK: 1000 samples, 30 epochs")
    print("="*80)
    print(f"\nTesting {len(models_to_test)} models on {len(datasets_to_test)} datasets")
    print(f"Models: {', '.join(models_to_test)}")
    print(f"Datasets: {', '.join(datasets_to_test)}\n")

    for dataset_name in datasets_to_test:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}\n")

        # Prepare data
        if dataset_name == "synthetic":
            full_ds = get_dataset(dataset_name, n_samples=1000, d_x=2, seed=42)
        else:
            full_ds = get_dataset(dataset_name, n_samples=1000, d_x=2, seed=42, label_ratio=0.5)

        half = len(full_ds) // 2
        train_ds = TensorDataset(*(t[:half] for t in full_ds.tensors))
        val_ds = TensorDataset(*(t[half:] for t in full_ds.tensors))

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)

        x_dim = train_ds.tensors[0].size(1)
        y_dim = train_ds.tensors[1].size(1)

        for model_name in models_to_test:
            print(f"\n{'─'*60}")
            print(f"Model: {model_name}")
            print(f"{'─'*60}")

            try:
                # Build model
                kwargs = {"d_x": x_dim, "d_y": y_dim, "k": 2}
                if model_name == "lp_knn":
                    kwargs["n_neighbors"] = 3
                elif model_name == "ctm_t":
                    kwargs = {"d_in": x_dim + y_dim + 1}

                model = get_model(model_name, **kwargs)

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

                # Train
                start_time = time.time()
                trainer = Trainer(model, opt, train_loader, val_loader=val_loader, logger=None)

                # Track progress every 10 epochs
                for epoch in range(3):  # 3 x 10 = 30 epochs
                    trainer.fit(10)
                    val_metrics = trainer.evaluate(val_loader)
                    print(f"  Epoch {(epoch+1)*10:2d}: "
                          f"val_rmse={val_metrics['outcome rmse']:.4f}, "
                          f"val_acc={val_metrics['treatment accuracy']:.2f}")

                train_time = time.time() - start_time

                # Final evaluation
                train_metrics = trainer.evaluate(train_loader)
                val_metrics = trainer.evaluate(val_loader)

                result = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "train_loss": train_metrics["loss"],
                    "train_rmse": train_metrics["outcome rmse"],
                    "train_acc": train_metrics["treatment accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_rmse": val_metrics["outcome rmse"],
                    "val_acc": val_metrics["treatment accuracy"],
                    "train_time_seconds": train_time,
                }

                results["results"].append(result)

                print(f"  ✓ Final: val_rmse={val_metrics['outcome rmse']:.4f}, "
                      f"val_acc={val_metrics['treatment accuracy']:.2f}, "
                      f"time={train_time:.1f}s")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                results["results"].append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "error": str(e)
                })

    # Save results
    output_file = "experimental_benchmark_1000_30.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_file}")

    # Print summary comparison
    print("\n" + "="*80)
    print("SUMMARY: Comparison with CI Baseline")
    print("="*80)

    for dataset_name in datasets_to_test:
        print(f"\n{dataset_name.upper()} Dataset:")
        print("-" * 60)
        print(f"{'Model':<25} {'CI (100/10)':<15} {'Exp (1000/30)':<15} {'Change':<15}")
        print("-" * 60)

        dataset_results = [r for r in results["results"] if r.get("dataset") == dataset_name and "error" not in r]
        for result in sorted(dataset_results, key=lambda x: x.get("val_rmse", 999)):
            model = result["model"]
            exp_rmse = result["val_rmse"]

            # These are approximate CI values - you'd load actual CI results for real comparison
            print(f"{model:<25} {'(varies)':<15} {exp_rmse:<15.4f} {'TBD':<15}")

    return results


if __name__ == "__main__":
    results = run_experiment()
