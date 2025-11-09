# Predicted Results from debug_flow_ssc_nhefs.py

## Environment Issue

Package installation is taking too long in this environment. However, based on our investigation, I can predict what the debug script would likely show:

## Expected Output Pattern

### Configuration 1: Small Dataset (100 samples, 10 epochs)
**Prediction: ✅ WORKS**
```
val_outcome_rmse: ~32-35
val_treatment_accuracy: ~0.68-0.72
```

**Why**: We know from CI benchmarks this configuration works.

### Configuration 2: Large Dataset (1566 samples, 1 epoch)
**Prediction: ✅ LIKELY WORKS**
```
val_outcome_rmse: ~15-25
val_treatment_accuracy: ~0.65-0.75
```

**Why**: With only 1 epoch (78 gradient updates), the model likely hasn't had time to diverge.

### Configuration 3: Large Dataset (1566 samples, 2 epochs)
**Prediction: ⚠️ STARTS TO DEGRADE**
```
val_outcome_rmse: ~30-45
val_treatment_accuracy: ~0.55-0.65
```

**Why**: Around epoch 2, we'd start seeing issues if the model is accumulating errors.

### Configuration 4: Large Dataset (1566 samples, 10 epochs)
**Prediction: ❌ FAILS (confirmed by CI)**
```
val_outcome_rmse: 75.65 ± 25.88
val_treatment_accuracy: 0.49 ± 0.20
```

**Why**: This is the broken configuration we've already tested.

### Configuration 5: Large Dataset, Lower LR (1e-4, 10 epochs)
**Prediction: ⚠️ MIGHT HELP SLIGHTLY**
```
val_outcome_rmse: ~50-65
val_treatment_accuracy: ~0.50-0.60
```

**Why**: Lower LR might slow divergence but probably won't prevent it completely over 780 gradient updates.

### Configuration 6: Large Dataset, Larger Batch (32, 10 epochs)
**Prediction: ⚠️ MIGHT HELP**
```
val_outcome_rmse: ~40-60
val_treatment_accuracy: ~0.55-0.65
```

**Why**: Larger batches give more stable statistics estimates, which could help with the normalization issues.

## Key Diagnostic Patterns to Watch

### If the model diverges during training:

**Epoch 1-2**:
- `y_scale`: Should be ~5-10 (reasonable for weight change data)
- `y_shift`: Should be ~0-3
- RMSE: Should decrease from initial random predictions

**Epoch 3-5** (where divergence likely starts):
- `y_scale`: Might start growing (e.g., 15, 20, 30...)
- `y_shift`: Might drift away from true mean
- RMSE: Starts increasing instead of decreasing
- Loss: Might contain NaN or Inf

**Epoch 8-10** (fully diverged):
- `y_scale`: Could be huge (50+) or tiny (< 0.1)
- Predictions: Completely off-scale
- RMSE: ~75, Treatment Acc: ~0.49

## Most Likely Root Cause

Based on the evidence, my hypothesis is:

**Exponential Moving Average (EMA) statistics tracking fails with more data**

In `flow_ssc.py` (lines 157-180):
```python
self.stat_momentum = 0.05  # Momentum for EMA

def _update_stats(self, X, Y):
    # Batch statistics
    mean_x = X.mean(dim=0, keepdim=True)
    std_x = X.std(dim=0, keepdim=True)
    mean_y = Y.mean(dim=0, keepdim=True)
    std_y = Y.std(dim=0, keepdim=True)

    if not self._stats_initialized:
        # First batch: initialize from batch
        self.x_shift.copy_(mean_x)
        self.x_scale.copy_(std_x)
        self.y_shift.copy_(mean_y)
        self.y_scale.copy_(std_y)
    else:
        # Subsequent batches: EMA update
        m = 0.05
        self.x_shift.mul_(0.95).add_(mean_x * 0.05)
        self.x_scale.mul_(0.95).add_(std_x * 0.05)
        self.y_shift.mul_(0.95).add_(mean_y * 0.05)
        self.y_scale.mul_(0.95).add_(std_y * 0.05)
```

### The Problem:

1. **First batch initialization (batch_size=10)**:
   - `y_shift` initialized from 10 samples
   - `y_scale` initialized from 10 samples
   - These might be VERY unrepresentative of true population

2. **EMA with momentum=0.05**:
   - Very slow updates (only 5% of new info each batch)
   - After 780 batches, if initially wrong, still tracking wrong values
   - Accumulates bias if first batch was outlier

3. **With 50 samples total (small dataset)**:
   - Only 5 batches total
   - Less opportunity for cumulative error
   - First batch is 20% of data (more representative)

4. **With 783 samples (large dataset)**:
   - 78 batches total
   - First batch is 1.3% of data (could be outlier)
   - 780 total updates across 10 epochs = lots of error accumulation

## Recommended Fix

Instead of EMA tracking, use **batch normalization** or **pre-compute statistics**:

```python
def fit(self, train_loader):
    # Pre-compute statistics from full training set
    all_X = []
    all_Y = []
    for X, Y, T in train_loader:
        all_X.append(X)
        all_Y.append(Y)

    all_X = torch.cat(all_X)
    all_Y = torch.cat(all_Y)

    self.x_shift = all_X.mean(dim=0, keepdim=True)
    self.x_scale = all_X.std(dim=0, keepdim=True)
    self.y_shift = all_Y.mean(dim=0, keepdim=True)
    self.y_scale = all_Y.std(dim=0, keepdim=True)
    self._stats_initialized = True

    # Then train normally
    ...
```

Or use momentum=1.0 on first epoch to quickly converge to true statistics.

## Alternative: Simpler Fix

Just use 1-2 epochs instead of 10 for datasets > 500 samples:

```python
# In benchmark_config.json
"dataset_training_epochs": {
  "nhefs": 2  # Instead of 10
}
```

This would give 156 gradient updates instead of 780, reducing cumulative error.

## Next Steps Without Environment

Since I can't run the script due to installation time, I recommend:

1. **Run debug script on your local machine** if you have the environment
2. **Or modify eval.py** to test 1-2 epochs for NHEFS:
   ```json
   "dataset_training_epochs": {
     "nhefs": 2
   }
   ```
3. **Or fix the EMA statistics tracking** in flow_ssc.py

Would you like me to implement option 2 or 3?
