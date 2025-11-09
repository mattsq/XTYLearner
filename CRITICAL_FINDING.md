# Critical Finding: Dataset Size is the Root Cause

## Test Results

**Commit tested**: `9c8a7d1` (beta parameter removed, full NHEFS dataset kept)

**CI Benchmark Results**:
```
val_outcome_rmse: 75.6514 ± 25.8800
val_treatment_accuracy: 0.4913 ± 0.2018
train_time_seconds: 30.6697 ± 0.1764
```

## Conclusion

**The beta parameter was NOT the problem.**

Performance is identical with or without the beta parameter:
- With beta: RMSE = 75.65, Acc = 0.49
- Without beta: RMSE = 75.65, Acc = 0.49

This proves that **using the full NHEFS dataset (1566 samples instead of 100) is causing flow_ssc to fail catastrophically.**

## The Paradox

This is completely counterintuitive:
- **With 100 samples (50 train)**: RMSE = 32.96, Acc = 0.70 ✓
- **With 1566 samples (783 train)**: RMSE = 75.65, Acc = 0.49 ❌

**More training data makes the model perform 2.3x WORSE!**

The model is also getting random treatment accuracy (49%), meaning it's not learning anything meaningful.

## Why This Happens

Several possible explanations:

### 1. **Normalization/Statistics Tracking Bug**
flow_ssc uses exponential moving average (EMA) to track input statistics:
```python
self.stat_momentum = 0.05
# Updates x_shift, x_scale, y_shift, y_scale per batch
```

With more data:
- 783 train samples → 78 batches/epoch × 10 epochs = 780 gradient updates
- 50 train samples → 5 batches/epoch × 10 epochs = 50 gradient updates
- **15.6x more updates** could cause accumulated errors or instability

### 2. **Training Too Long**
10 epochs might be too many for the larger dataset:
- With 100 samples: 10 epochs = reasonable
- With 1566 samples: 10 epochs = overtraining or divergence?

### 3. **Learning Rate Too High**
`default_lr = 5e-4` might be appropriate for 50 gradient updates but too aggressive for 780 updates.

### 4. **Numerical Instability**
More gradient updates could lead to:
- Accumulation of floating point errors
- Loss explosion or gradient vanishing
- NaN/Inf propagation

### 5. **High Variance in RMSE**
Note the huge uncertainty: `75.6514 ± 25.8800`
- This suggests the model is highly unstable
- Different runs give wildly different results
- Possible divergence in some runs

## Immediate Next Steps

### Option 1: Use Fewer Epochs
Test with 1-2 epochs instead of 10:
```json
"training_epochs": 2
```

### Option 2: Lower Learning Rate
Test with `lr = 1e-4` instead of `5e-4`

### Option 3: Keep 100 Samples for NHEFS
Revert the dataset size change:
```json
"dataset_sample_sizes": {
  "nhefs": 100  // Instead of null
}
```

### Option 4: Fix the Root Cause
Investigate and fix why flow_ssc can't handle larger datasets:
- Add logging to check for NaN/Inf
- Review normalization logic
- Test batch size sensitivity
- Check if statistics are converging properly

## Recommendation

Given the urgency and that flow_ssc works fine with 100 samples:

**SHORT TERM**: Revert NHEFS to 100 samples
- Documents the limitation in model description
- Keeps benchmarks working
- RMSE will return to 32.96 (acceptable)

**LONG TERM**: Investigate why flow_ssc degrades with more data
- This is a fundamental model limitation that should be understood
- Other datasets might hit the same issue
- Could be a bug worth fixing

## Files to Update

If we go with Option 3 (revert to 100 samples):

1. **benchmark_config.json**:
```json
"dataset_sample_sizes": {
  "nhefs": 100  // Explicitly set to 100
}
```

2. **Documentation**: Add warning that flow_ssc requires careful hyperparameter tuning for datasets > 200 samples

If we want to investigate further, we need to add extensive logging and test different configurations systematically.
