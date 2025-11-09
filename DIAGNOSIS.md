# flow_ssc NHEFS Performance Degradation - Diagnosis

## Problem Summary

After implementing improvements to flow_ssc and correcting NHEFS benchmarking:
- **Expected**: RMSE improves from 32.96 → ~11.15 (66% improvement)
- **Actual**: RMSE degraded from 32.96 → 75.65 (2.3x WORSE!)
- **Treatment accuracy**: Degraded from 0.70 → 0.49 (random guessing)

## Timeline of Changes

| Commit | Change | NHEFS Result |
|--------|--------|--------------|
| 7f08292 | Baseline (no beta, 100 samples) | RMSE=32.96, Acc=0.70 ✓ |
| 80b7313 | Added beta parameter (default=1.0) | Not benchmarked |
| 8d202f1 | Added adaptive_beta (default=False) | Not benchmarked  |
| fa78154 | Changed to full dataset (1566 samples) | Not benchmarked |
| 05fe8ac | All changes merged | RMSE=75.65, Acc=0.49 ❌ |

## Investigation Results

### ✅ Step 1: Beta Parameter Configuration
- Default `beta = 1.0` ✓
- Default `adaptive_beta = False` ✓
- eval.py doesn't override these ✓
- With these defaults, the loss formula should be mathematically identical to the original

### ✅ Step 2: Data Preparation
- `n_samples = None` correctly loads full 1566 samples ✓
- Train/val split: 783/783 samples ✓
- Batch size: 10 ✓
- Data preparation logic looks correct ✓

### ⚠️ Critical Findings

**The beta parameter was NEVER tested alone!**
- No benchmark results exist for commits 80b7313, 8d202f1, 9196242
- We don't know if the beta parameter or dataset size (or their combination) caused the issue

**Mathematical equivalence unclear:**
- Old: `loss_lab = -(ll_x + ll_y).mean() + ce_clf`
- New: `loss_lab = -(effective_beta * ll_x + ll_y).mean() + ce_clf`
- With `effective_beta = 1.0`, these SHOULD be identical
- But could there be a numerical precision issue?

## Hypotheses

### Hypothesis 1: Beta Parameter Breaks Something
Even though beta=1.0 should be transparent, maybe:
- PyTorch treats `1.0 * ll_x` differently than `ll_x`
- There's a gradient flow issue
- The multiplication introduces numerical instability

### Hypothesis 2: Dataset Size Alone
The larger dataset (783 vs 50 train samples) might:
- Cause different normalization behavior with EMA stats
- Lead to numerical overflow/underflow
- Require different hyperparameters (epochs, learning rate)

### Hypothesis 3: Combination of Both
The beta parameter + larger dataset together cause:
- Accumulated numerical errors over 15.6x more gradient updates
- Statistics tracking issues
- Optimization landscape changes

## Next Steps

1. **Test beta parameter alone** (with 100 samples):
   - Revert to 100 samples
   - Keep beta parameter
   - See if RMSE stays at 32.96 or degrades

2. **Test dataset size alone** (without beta):
   - Use 1566 samples
   - Revert beta parameter
   - See if RMSE improves to 11.15 or degrades

3. **Check for NaN/Inf**:
   - Add logging to see if loss contains NaN/Inf
   - Check if model diverges during training

4. **Test with adjusted hyperparameters**:
   - Fewer epochs (1-2 instead of 10)
   - Lower learning rate (1e-4 instead of 5e-4)

## Recommendation

**Revert the beta parameter changes** and test with just the dataset size increase to isolate the root cause.
