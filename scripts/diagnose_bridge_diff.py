"""Diagnostic script to analyze BridgeDiff training behavior."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
from pathlib import Path

from xtylearner.models.bridge_diff import BridgeDiff
from xtylearner.data import load_mixed_synthetic_dataset, load_toy_dataset
from xtylearner.training import Trainer


def analyze_loss_components(model, loader, n_batches=10):
    """Analyze loss components over several batches."""
    model.eval()

    loss_obs_list = []
    loss_unobs_list = []
    ce_loss_list = []
    total_loss_list = []

    with torch.no_grad():
        for i, (x, y, t) in enumerate(loader):
            if i >= n_batches:
                break

            # Compute loss components separately
            b = x.size(0)
            device = x.device
            tau = torch.rand(b, device=device)
            sigma = model._sigma(tau).unsqueeze(-1)
            eps = torch.randn_like(y)
            y_noisy = y + sigma * eps

            logits = model.classifier(x, y)

            # Observed loss
            obs_mask = t != -1
            loss_obs = torch.tensor(0.0)
            if obs_mask.any():
                s_obs = model.score_net(
                    y_noisy[obs_mask],
                    x[obs_mask],
                    t[obs_mask].clamp_min(0),
                    tau[obs_mask],
                )
                sig_obs = sigma[obs_mask].clamp_min(1e-6)
                inv_sig = sig_obs.reciprocal()
                mse = (s_obs + eps[obs_mask] * inv_sig) ** 2
                loss_obs = (sig_obs**2 * mse).mean()

            # Unobserved loss
            unobs_mask = obs_mask.logical_not()
            loss_unobs = torch.tensor(0.0)
            if unobs_mask.any():
                sig_unobs = sigma[unobs_mask].clamp_min(1e-6)
                eps_unobs = eps[unobs_mask]
                tau_unobs = tau[unobs_mask]
                x_unobs = x[unobs_mask]
                y_unobs = y_noisy[unobs_mask]

                x_rep = x_unobs.repeat_interleave(model.k, dim=0)
                y_rep = y_unobs.repeat_interleave(model.k, dim=0)
                tau_rep = tau_unobs.repeat_interleave(model.k, dim=0)
                t_vals = torch.arange(model.k, device=device).repeat(x_unobs.size(0))

                scores = model.score_net(y_rep, x_rep, t_vals, tau_rep)
                scores = scores.view(x_unobs.size(0), model.k, model.d_y)

                p_post = torch.softmax(logits[unobs_mask], dim=-1)
                inv_sig = sig_unobs.reciprocal().unsqueeze(1)
                eps_term = eps_unobs.unsqueeze(1) * inv_sig
                mse = (scores + eps_term) ** 2
                mse = mse.mean(dim=-1)

                weight = (sig_unobs.squeeze(-1).unsqueeze(1) ** 2)
                loss_unobs = (p_post * weight * mse).sum(dim=1).mean()

            # CE loss
            ce_loss = torch.tensor(0.0)
            if obs_mask.any():
                ce_loss = torch.nn.functional.cross_entropy(
                    logits[obs_mask], t[obs_mask].clamp_min(0)
                )

            loss_obs_list.append(loss_obs.item())
            loss_unobs_list.append(loss_unobs.item())
            ce_loss_list.append(ce_loss.item())
            total_loss_list.append(loss_obs.item() + loss_unobs.item() + ce_loss.item())

    return {
        "loss_obs": loss_obs_list,
        "loss_unobs": loss_unobs_list,
        "ce_loss": ce_loss_list,
        "total_loss": total_loss_list,
    }


def train_with_monitoring(dataset_name="synthetic", n_samples=100, epochs=20):
    """Train bridge_diff with detailed monitoring."""

    print(f"\n{'='*60}")
    print(f"Training BridgeDiff on {dataset_name} dataset")
    print(f"{'='*60}\n")

    # Load data
    if dataset_name == "synthetic":
        dataset = load_toy_dataset(n_samples=n_samples, d_x=2, seed=42)
    else:
        dataset = load_mixed_synthetic_dataset(
            n_samples=n_samples, d_x=2, seed=42, label_ratio=0.5
        )

    half = len(dataset) // 2
    from torch.utils.data import TensorDataset
    train_ds = TensorDataset(*(t[:half] for t in dataset.tensors))
    val_ds = TensorDataset(*(t[half:] for t in dataset.tensors))

    train_loader = DataLoader(train_ds, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=10)

    # Create model
    model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=64, hidden=256, n_blocks=3)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training history
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_rmse": [],
        "val_rmse": [],
        "train_acc": [],
        "val_acc": [],
        "loss_components": []
    }

    trainer = Trainer(model, opt, train_loader, val_loader=val_loader, logger=None)

    # Train with monitoring
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train one epoch
        trainer.fit(1)

        # Evaluate
        train_metrics = trainer.evaluate(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        # Analyze loss components
        components = analyze_loss_components(model, train_loader, n_batches=5)

        # Record
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics.get("loss", 0))
        history["val_loss"].append(val_metrics.get("loss", 0))
        history["train_rmse"].append(train_metrics.get("outcome rmse", 0))
        history["val_rmse"].append(val_metrics.get("outcome rmse", 0))
        history["train_acc"].append(train_metrics.get("treatment accuracy", 0))
        history["val_acc"].append(val_metrics.get("treatment accuracy", 0))
        history["loss_components"].append({
            "epoch": epoch + 1,
            "loss_obs_mean": np.mean(components["loss_obs"]),
            "loss_obs_std": np.std(components["loss_obs"]),
            "loss_unobs_mean": np.mean(components["loss_unobs"]),
            "loss_unobs_std": np.std(components["loss_unobs"]),
            "ce_loss_mean": np.mean(components["ce_loss"]),
            "ce_loss_std": np.std(components["ce_loss"]),
        })

        # Print summary
        print(f"  Train: loss={train_metrics['loss']:.4f}, "
              f"rmse={train_metrics['outcome rmse']:.4f}, "
              f"acc={train_metrics['treatment accuracy']:.2f}")
        print(f"  Val:   loss={val_metrics['loss']:.4f}, "
              f"rmse={val_metrics['outcome rmse']:.4f}, "
              f"acc={val_metrics['treatment accuracy']:.2f}")
        print(f"  Loss components: "
              f"obs={np.mean(components['loss_obs']):.4f}, "
              f"unobs={np.mean(components['loss_unobs']):.4f}, "
              f"ce={np.mean(components['ce_loss']):.4f}")

    return history, trainer


def plot_training_curves(history, output_path="bridge_diff_training_curves.png"):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = history["epoch"]

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train", marker='o')
    axes[0, 0].plot(epochs, history["val_loss"], label="Val", marker='s')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # RMSE
    axes[0, 1].plot(epochs, history["train_rmse"], label="Train", marker='o')
    axes[0, 1].plot(epochs, history["val_rmse"], label="Val", marker='s')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].set_title("Outcome RMSE")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Accuracy
    axes[1, 0].plot(epochs, history["train_acc"], label="Train", marker='o')
    axes[1, 0].plot(epochs, history["val_acc"], label="Val", marker='s')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Treatment Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.1])

    # Loss components
    loss_obs_means = [c["loss_obs_mean"] for c in history["loss_components"]]
    loss_unobs_means = [c["loss_unobs_mean"] for c in history["loss_components"]]
    ce_loss_means = [c["ce_loss_mean"] for c in history["loss_components"]]

    axes[1, 1].plot(epochs, loss_obs_means, label="Observed", marker='o')
    axes[1, 1].plot(epochs, loss_unobs_means, label="Unobserved", marker='s')
    axes[1, 1].plot(epochs, ce_loss_means, label="CE", marker='^')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Loss Components")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to {output_path}")


def main():
    """Run comprehensive diagnostics."""

    # Train on synthetic dataset
    history_syn, trainer_syn = train_with_monitoring("synthetic", n_samples=100, epochs=30)

    # Train on synthetic_mixed dataset
    history_mixed, trainer_mixed = train_with_monitoring("synthetic_mixed", n_samples=100, epochs=30)

    # Save results
    results = {
        "synthetic": history_syn,
        "synthetic_mixed": history_mixed,
    }

    with open("bridge_diff_diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Diagnostic Results Summary")
    print(f"{'='*60}\n")

    print("Synthetic Dataset:")
    print(f"  Final Train RMSE: {history_syn['train_rmse'][-1]:.4f}")
    print(f"  Final Val RMSE: {history_syn['val_rmse'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history_syn['val_acc'][-1]:.2f}")

    print("\nSynthetic Mixed Dataset:")
    print(f"  Final Train RMSE: {history_mixed['train_rmse'][-1]:.4f}")
    print(f"  Final Val RMSE: {history_mixed['val_rmse'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history_mixed['val_acc'][-1]:.2f}")

    print(f"\n{'='*60}")
    print("Comparison with CI Baseline:")
    print(f"{'='*60}\n")
    print("Expected (from CI):")
    print("  Synthetic val RMSE: 2.45512")
    print("  Synthetic_mixed val RMSE: 1.67121")

    print("\nObserved (from this run):")
    print(f"  Synthetic val RMSE: {history_syn['val_rmse'][-1]:.4f}")
    print(f"  Synthetic_mixed val RMSE: {history_mixed['val_rmse'][-1]:.4f}")

    # Plot training curves
    plot_training_curves(history_syn, "bridge_diff_synthetic_curves.png")
    plot_training_curves(history_mixed, "bridge_diff_mixed_curves.png")

    print("\nDiagnostic complete! Check:")
    print("  - bridge_diff_diagnostic_results.json")
    print("  - bridge_diff_synthetic_curves.png")
    print("  - bridge_diff_mixed_curves.png")


if __name__ == "__main__":
    main()
