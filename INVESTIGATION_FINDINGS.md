# Investigation Findings - flow_ssc NHEFS Performance Issue

## Executive Summary

I followed your plan to determine the source of the flow_ssc NHEFS performance degradation (RMSE 32.96 → 75.65). After systematic investigation, I identified that **we don't have enough data to determine the exact cause** because the beta parameter change was never benchmarked independently.

## What I Found

### ✅ Step 1: Verified beta parameter configuration

**Result**: Configuration is correct
- `beta = 1.0` (default) ✓
- `adaptive_beta = False` (default) ✓
- No config files override these values ✓

The beta parameter, as configured, should be mathematically transparent:
- Old formula: `-(ll_x + ll_y).mean()`
- New formula: `-(1.0 * ll_x + ll_y).mean()`
- These should be identical

### ✅ Step 2: Analyzed data split and normalization

**Result**: Data preparation looks correct
- `n_samples = None` correctly loads full 1566 samples ✓
- Train/val split: 783/783 samples ✓
- Batch size: 10 ✓
- X standardization in dataset: ✓

### 🔍 Critical Discovery: Timeline Gap

The CI benchmark history reveals:

| Version | NHEFS Samples | Beta Parameter | RMSE | Accuracy | Benchmarked? |
|---------|--------------|----------------|------|----------|--------------|
| 7f08292 | 100 (50 train) | No | 32.96 | 0.70 | ✅ Yes |
| 80b7313 | 100 (50 train) | Yes (beta=1.0) | ? | ? | ❌ **NO** |
| 8d202f1 | 100 (50 train) | Yes + adaptive | ? | ? | ❌ **NO** |
| fa78154 | 1566 (783 train) | Yes + adaptive | ? | ? | ❌ **NO** |
| 05fe8ac | 1566 (783 train) | Yes + adaptive | 75.65 | 0.49 | ✅ Yes |

**The problem**: We can't determine which change caused the degradation because intermediate versions were never benchmarked!

## Possible Causes

### Hypothesis 1: Beta Parameter Issue
Even though beta=1.0 should be transparent, maybe:
- PyTorch handles `1.0 * tensor` differently than `tensor`
- There's a subtle numerical precision issue
- Gradient computation is affected

### Hypothesis 2: Dataset Size Issue
The larger dataset (783 vs 50 train samples) might:
- Expose bugs in normalization/statistics tracking
- Require different hyperparameters (learning rate, epochs)
- Cause numerical instability over 15.6x more gradient updates

### Hypothesis 3: Combination
Both changes together create an unexpected interaction.

## What I Did - Step 4: Isolated the Issue

I **reverted the beta parameter changes** while **keeping the full NHEFS dataset** to test which change is responsible.

**Commit**: `9c8a7d1` - "Revert beta parameter to isolate NHEFS performance issue"

**Changes**:
- `xtylearner/models/flow_ssc.py`: Reverted to pre-beta version (from 7f08292)
- NHEFS still uses full 1566 samples (from fa78154)
- All other changes unchanged

## Expected Outcomes from CI

### If Beta Parameter Was the Problem:
- NHEFS RMSE will improve to ~11-15 (good!)
- Treatment accuracy will improve to ~0.70-0.85
- **Action**: Keep dataset size change, remove beta parameter permanently

### If Dataset Size Was the Problem:
- NHEFS RMSE will still be bad (~75.65)
- Treatment accuracy will still be poor (~0.49)
- **Action**: Investigate why larger dataset breaks the model (hyperparameters? normalization?)

### If Combination Was the Problem:
- NHEFS RMSE might be intermediate (e.g., 40-50)
- **Action**: Need to test beta parameter with small dataset separately

## Next Steps

1. **Wait for CI benchmark results** on commit `9c8a7d1`
   - This will tell us if the beta parameter or dataset size is the issue

2. **If beta parameter was the issue**:
   - Remove beta/adaptive_beta from flow_ssc.py permanently
   - Document that the mathematical equivalence doesn't hold in practice
   - Investigate why `1.0 * ll_x` differs from `ll_x`

3. **If dataset size was the issue**:
   - Investigate normalization with larger datasets
   - Test with adjusted hyperparameters (fewer epochs, lower learning rate)
   - Check for NaN/Inf in loss values
   - Consider if flow_ssc fundamentally doesn't scale to larger datasets

4. **If still unclear**:
   - Test beta parameter with 100 samples (no dataset change)
   - Add extensive logging to understand what's happening during training

## Files Created

- `DIAGNOSIS.md`: Detailed technical investigation notes
- `INVESTIGATION_FINDINGS.md`: This file - user-friendly summary
- `test_flow_ssc_nhefs.py`: Test script for debugging

## Recommendation

Please run the CI benchmarks on the current branch (`claude/improve-flow-scc-nhefs-011CUwjBZYv6B6BtfxE757r3`) to get the results for commit `9c8a7d1`. Once we have those results, we'll know definitively whether the beta parameter or dataset size (or both) is causing the problem.
