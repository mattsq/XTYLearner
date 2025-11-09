"""Debug script to investigate flow_ssc NHEFS scaling issue.

This script trains flow_ssc on different NHEFS configurations and logs
detailed diagnostics to understand why performance degrades with more data.
"""

import torch
import numpy as np
from xtylearner.data import get_dataset
from xtylearner.models import get_model
from xtylearner.training import Trainer
from torch.utils.data import DataLoader, TensorDataset


def test_configuration(n_samples, epochs, lr, batch_size, name):
    """Test a specific configuration and return diagnostics."""
    print(f"\n{'='*70}")
    print(f"Configuration: {name}")
    print(f"{'='*70}")
    print(f"  n_samples: {n_samples}")
    print(f"  epochs: {epochs}")
    print(f"  learning_rate: {lr}")
    print(f"  batch_size: {batch_size}")

    # Load dataset
    if n_samples == "full":
        full_ds = get_dataset("nhefs", n_samples=None, seed=42)
        n_samples_actual = len(full_ds)
    else:
        full_ds = get_dataset("nhefs", n_samples=n_samples, seed=42)
        n_samples_actual = len(full_ds)

    half = len(full_ds) // 2
    train_ds = TensorDataset(*(t[:half] for t in full_ds.tensors))
    val_ds = TensorDataset(*(t[half:] for t in full_ds.tensors))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    n_train = len(train_ds)
    n_batches = len(train_loader)
    total_updates = n_batches * epochs

    print(f"\nDataset info:")
    print(f"  Total samples: {n_samples_actual}")
    print(f"  Train samples: {n_train}")
    print(f"  Val samples: {len(val_ds)}")
    print(f"  Features (d_x): {train_ds.tensors[0].shape[1]}")
    print(f"  Batches per epoch: {n_batches}")
    print(f"  Total gradient updates: {total_updates}")

    # Create model
    d_x = train_ds.tensors[0].shape[1]
    model = get_model("flow_ssc", d_x=d_x, d_y=1, k=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Custom trainer to log epoch-level stats
    trainer = Trainer(model, optimizer, train_loader, val_loader=val_loader, logger=None)

    # Track stats per epoch
    epoch_stats = []

    print(f"\nTraining...")
    for epoch in range(epochs):
        # Train one epoch
        trainer.fit(1)

        # Evaluate
        val_metrics = trainer.evaluate(val_loader)

        # Get model stats if available
        stats = {
            'epoch': epoch + 1,
            'val_outcome_rmse': val_metrics.get('outcome rmse', float('nan')),
            'val_treatment_accuracy': val_metrics.get('treatment accuracy', float('nan')),
        }

        # Add normalization statistics
        with torch.no_grad():
            stats.update({
                'x_shift_mean': model.x_shift.mean().item(),
                'x_shift_std': model.x_shift.std().item(),
                'x_scale_mean': model.x_scale.mean().item(),
                'x_scale_std': model.x_scale.std().item(),
                'y_shift': model.y_shift.item(),
                'y_scale': model.y_scale.item(),
            })

        epoch_stats.append(stats)

        print(f"  Epoch {epoch+1}/{epochs}: "
              f"val_rmse={stats['val_outcome_rmse']:.4f}, "
              f"val_acc={stats['val_treatment_accuracy']:.4f}, "
              f"y_scale={stats['y_scale']:.4f}")

    # Final evaluation
    final_metrics = trainer.evaluate(val_loader)

    print(f"\nFinal Results:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nNormalization Statistics (final):")
    with torch.no_grad():
        print(f"  x_shift: mean={model.x_shift.mean():.4f}, std={model.x_shift.std():.4f}")
        print(f"  x_scale: mean={model.x_scale.mean():.4f}, std={model.x_scale.std():.4f}")
        print(f"  y_shift: {model.y_shift.item():.4f}")
        print(f"  y_scale: {model.y_scale.item():.4f}")

    return {
        'config': name,
        'n_samples': n_samples_actual,
        'n_train': n_train,
        'total_updates': total_updates,
        'final_metrics': final_metrics,
        'epoch_stats': epoch_stats,
    }


def main():
    """Run diagnostic tests."""

    print("="*70)
    print("flow_ssc NHEFS Scaling Investigation")
    print("="*70)

    results = []

    # Configuration 1: Small dataset (known to work)
    results.append(test_configuration(
        n_samples=100,
        epochs=10,
        lr=5e-4,
        batch_size=10,
        name="Small dataset (baseline - should work)"
    ))

    # Configuration 2: Large dataset, 1 epoch only
    results.append(test_configuration(
        n_samples="full",
        epochs=1,
        lr=5e-4,
        batch_size=10,
        name="Large dataset, 1 epoch"
    ))

    # Configuration 3: Large dataset, 2 epochs
    results.append(test_configuration(
        n_samples="full",
        epochs=2,
        lr=5e-4,
        batch_size=10,
        name="Large dataset, 2 epochs"
    ))

    # Configuration 4: Large dataset, 10 epochs (broken)
    results.append(test_configuration(
        n_samples="full",
        epochs=10,
        lr=5e-4,
        batch_size=10,
        name="Large dataset, 10 epochs (broken config)"
    ))

    # Configuration 5: Large dataset, lower LR
    results.append(test_configuration(
        n_samples="full",
        epochs=10,
        lr=1e-4,
        batch_size=10,
        name="Large dataset, lower LR"
    ))

    # Configuration 6: Large dataset, larger batch size
    results.append(test_configuration(
        n_samples="full",
        epochs=10,
        lr=5e-4,
        batch_size=32,
        name="Large dataset, larger batch"
    ))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Configuration':<40} {'Updates':>10} {'RMSE':>10} {'Acc':>8}")
    print("-"*70)

    for r in results:
        rmse = r['final_metrics'].get('outcome rmse', float('nan'))
        acc = r['final_metrics'].get('treatment accuracy', float('nan'))
        print(f"{r['config']:<40} {r['total_updates']:>10} {rmse:>10.2f} {acc:>8.4f}")

    print("\nLook for patterns:")
    print("  - Does RMSE get worse with more epochs?")
    print("  - Does lower LR help?")
    print("  - Does larger batch size help?")
    print("  - When does the model start to fail?")


if __name__ == "__main__":
    main()
