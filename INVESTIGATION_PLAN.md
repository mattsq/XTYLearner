# Systematic Investigation Plan: flow_ssc Scaling Issue

## The Mystery

flow_ssc performance **degrades** with more training data:
- **50 train samples**: RMSE = 32.96, Acc = 0.70 ✓
- **783 train samples**: RMSE = 75.65, Acc = 0.49 ❌

Treatment accuracy of 0.49 = random guessing → model isn't learning at all.

## Key Questions to Answer

### 1. Does the loss diverge?
- Does loss become NaN or Inf during training?
- Does it explode to huge values?
- When does divergence happen (which epoch/batch)?

### 2. What happens to gradients?
- Are gradients vanishing (→0)?
- Are gradients exploding (→∞)?
- Do they contain NaN?

### 3. What about normalization statistics?
- `x_shift`, `x_scale`, `y_shift`, `y_scale` tracked via EMA
- Do these converge to reasonable values?
- Are they stable across epochs?
- Does batch size (10) give unreliable statistics?

### 4. Is it a hyperparameter issue?
- **Epochs**: 10 epochs × 78 batches = 780 updates (vs 50 with small dataset)
- **Learning rate**: 5e-4 might be too high for 780 updates
- **Batch size**: 10 samples might be too small for 65-dimensional data

### 5. What do successful models do differently?
Other models work fine on full NHEFS:
- ganite: RMSE = 4.73
- mean_teacher: RMSE = 5.15
- prob_circuit: RMSE = 0.00

What do they do that flow_ssc doesn't?

## Investigation Phase 1: Diagnosis

### Step 1.1: Add Extensive Logging

Add to `flow_ssc.py`:
```python
def loss(self, X, Y, T_obs):
    # ... existing code ...

    # Log diagnostics
    if self.training:
        with torch.no_grad():
            # Log loss components
            if hasattr(self, 'loss_breakdown'):
                self.loss_breakdown = {
                    'loss_lab': loss_lab.item(),
                    'loss_ulb': loss_ulb.item(),
                    'll_x_mean': lp_x.mean().item(),
                    'll_x_std': lp_x.std().item(),
                    'll_y_mean': (ll_y.mean().item() if t_lab_mask.any() else 0),
                    'ce_clf': (ce_clf.item() if t_lab_mask.any() else 0),
                    # Statistics
                    'x_shift_mean': self.x_shift.mean().item(),
                    'x_scale_mean': self.x_scale.mean().item(),
                    'y_shift_mean': self.y_shift.mean().item(),
                    'y_scale_mean': self.y_scale.mean().item(),
                    # Check for NaN
                    'has_nan': bool(torch.isnan(loss_lab) or torch.isnan(loss_ulb)),
                    'has_inf': bool(torch.isinf(loss_lab) or torch.isinf(loss_ulb)),
                }
```

### Step 1.2: Create Debug Script

Create `debug_flow_ssc_nhefs.py`:
```python
"""Debug script to understand flow_ssc failure on large NHEFS."""

import torch
import numpy as np
from xtylearner.data import get_dataset
from xtylearner.models import get_model
from xtylearner.training import Trainer
from torch.utils.data import DataLoader, TensorDataset

# Load full NHEFS
full_ds = get_dataset("nhefs", n_samples=None, seed=42)
half = len(full_ds) // 2
train_ds = TensorDataset(*(t[:half] for t in full_ds.tensors))
val_ds = TensorDataset(*(t[half:] for t in full_ds.tensors))

print(f"Dataset sizes:")
print(f"  Train: {len(train_ds)}")
print(f"  Val: {len(val_ds)}")
print(f"  Features: {train_ds.tensors[0].shape[1]}")
print()

# Create model
model = get_model("flow_ssc", d_x=65, d_y=1, k=2)
print(f"Model config:")
print(f"  default_lr: {model.default_lr}")
print(f"  stat_momentum: {model.stat_momentum}")
print()

# Create loaders with different batch sizes to test
for batch_size in [10, 32, 64]:
    print(f"\n{'='*60}")
    print(f"Testing with batch_size={batch_size}")
    print(f"{'='*60}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Batches per epoch: {len(train_loader)}")

    # Train for just 1 epoch to start
    model_fresh = get_model("flow_ssc", d_x=65, d_y=1, k=2)
    optimizer = torch.optim.Adam(model_fresh.parameters(), lr=5e-4)
    trainer = Trainer(model_fresh, optimizer, train_loader, val_loader=val_loader)

    # Train 1 epoch and check
    print("\nTraining for 1 epoch...")
    trainer.fit(1)

    metrics = trainer.evaluate(val_loader)
    print(f"\nAfter 1 epoch:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Check for issues
    if hasattr(model_fresh, 'loss_breakdown'):
        print(f"\nLoss breakdown:")
        for k, v in model_fresh.loss_breakdown.items():
            print(f"  {k}: {v}")
```

### Step 1.3: Test Configurations

Test matrix:

| Epochs | LR | Batch Size | Description |
|--------|-----|-----------|-------------|
| 1 | 5e-4 | 10 | Baseline - 1 epoch only |
| 2 | 5e-4 | 10 | Check if diverges in epoch 2 |
| 10 | 5e-4 | 10 | Current broken config |
| 10 | 1e-4 | 10 | Lower LR |
| 10 | 5e-4 | 32 | Larger batch size |
| 10 | 5e-4 | 64 | Even larger batch |

## Investigation Phase 2: Comparison

### Step 2.1: Compare with Working Models

Run same debug script for models that work:
- `mean_teacher` (RMSE 5.15)
- `ganite` (RMSE 4.73)

Observe:
- How do their losses evolve?
- Do they use different batch sizes?
- Different learning rates?
- Different normalization strategies?

### Step 2.2: Compare with Small NHEFS

Run flow_ssc with 100 samples (known to work):
- Observe loss evolution
- Check statistics convergence
- Compare with 783 sample run

## Investigation Phase 3: Hypothesis Testing

Based on Phase 1-2 findings, test specific hypotheses:

### Hypothesis A: Too Many Gradient Updates
**Test**: Train with 1-2 epochs only
**Expected**: If this is the issue, model should work better

### Hypothesis B: Learning Rate Too High
**Test**: Train with lr=1e-4 or 1e-5
**Expected**: More stable training, better convergence

### Hypothesis C: Batch Size Too Small
**Test**: Train with batch_size=32 or 64
**Expected**: More stable statistics, better performance

### Hypothesis D: Statistics Tracking Bug
**Test**: Pre-compute statistics on full dataset instead of EMA
**Expected**: If EMA is the issue, this should work

### Hypothesis E: Numerical Precision
**Test**: Use float64 instead of float32
**Expected**: If precision is the issue, results improve

## Success Criteria

We'll know we've found the root cause when:
1. We can predict when flow_ssc will fail based on data characteristics
2. We have a configuration that works on full NHEFS (RMSE < 15)
3. We understand the failure mechanism well enough to document or fix it

## Next Steps

Let me know which phase you'd like to start with:
1. **Phase 1**: Add logging and run diagnostic tests
2. **Phase 2**: Compare with working models
3. **Phase 3**: Test specific hypotheses

Or we can do all phases systematically.
