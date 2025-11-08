

# flow_ssc NHEFS Performance Analysis

## Problem Summary

flow_ssc ranks **dead last** on the NHEFS dataset with val RMSE of 32.96, while most models achieve 5.7-6.5. This represents a **136.1x performance degradation** from synthetic to NHEFS - the worst among all 39 models tested.

## Root Cause

**Small sample size + High dimensionality = flow_ssc failure**

### Dataset Characteristics:

| Dataset | Samples (train) | Features (d_x) | flow_ssc RMSE | Rank |
|---------|----------------|----------------|---------------|------|
| Synthetic | 50 | 2 | 0.24 | 10th / 35+ |
| Synthetic Mixed | 50 | 2 | 0.62 | 10th / 35+ |
| **NHEFS** | **50** | **65** | **32.96** | **39th / 39** ⚠️ |

### Why flow_ssc Fails on NHEFS:

1. **flow_x needs to learn p(x) in 65 dimensions**
   - Requires estimating high-dimensional density
   - With only 50 samples, this is statistically impossible
   - flow_x adds noise instead of useful signal

2. **flow_y is conditioned on 67 dimensions (65 + 2 for treatment)**
   - Still very high-dimensional
   - But more tractable than modeling p(x)

3. **Performance degradation comparison**:
   - mean_teacher: 31.1x degradation (synthetic → NHEFS)
   - cycle_dual: 37.9x degradation
   - prob_circuit: 89.3x degradation
   - **flow_ssc: 136.1x degradation** ← WORST

## Solution: Beta Parameter

The **beta parameter** controls the weight of log p(x) in the loss:

```python
loss_lab = -(self.beta * ll_x + ll_y).mean() + ce_clf
loss_ulb = -(self.beta * lp_x_u + lse).mean()
```

### Results with different beta values on NHEFS:

| Beta | Val RMSE | Improvement | Interpretation |
|------|----------|-------------|----------------|
| 1.0  | 35.97 | baseline | Full flow_x (original) |
| 0.5  | 27.87 | +23% | Reduce flow_x weight |
| 0.1  | 25.14 | +30% | Mostly disable flow_x |
| **0.0** | **20.64** | **+37%** | Completely disable flow_x ✓ |

**Setting beta=0.0 improves NHEFS from 32.96 → 20.64 (37% improvement)**

However, 20.64 is still much worse than other models (5.7-6.5), indicating flow_ssc is fundamentally not suited for tiny high-dimensional datasets.

## Recommendations

### 1. **Document the limitation**
flow_ssc is designed for datasets with:
- Sample size > 200 (preferably 500+)
- Moderate dimensionality (d_x < 50)
- For small/high-d datasets, use simpler models (mean_teacher, fixmatch, etc.)

### 2. **Adaptive beta parameter**
Automatically adjust beta based on sample size and dimensionality:

```python
def compute_adaptive_beta(n_samples, d_x):
    """Compute beta based on statistical sufficiency."""
    # Rule of thumb: need ~10 samples per dimension for density estimation
    samples_per_dim = n_samples / d_x

    if samples_per_dim < 5:
        return 0.0  # Too few samples, disable flow_x
    elif samples_per_dim < 10:
        return 0.1  # Barely sufficient, down-weight heavily
    elif samples_per_dim < 20:
        return 0.5  # Moderate, use partial weight
    else:
        return 1.0  # Sufficient, use full model
```

For NHEFS: n_samples=50, d_x=65 → samples_per_dim=0.77 → **beta=0.0** ✓

### 3. **Skip NHEFS in benchmarks OR use adaptive beta**

Either:
- **Option A**: Exclude NHEFS from flow_ssc benchmarks (document why)
- **Option B**: Use adaptive beta so flow_ssc automatically adapts
- **Option C**: Set default beta=0.0 and document as "conditional-only mode"

### 4. **Model Variants**

Consider creating two variants:
- `flow_ssc` (full model, beta=1.0) - for large datasets
- `flow_ssc_lite` (conditional only, beta=0.0) - for small datasets

## Performance Comparison

With beta=0.0, flow_ssc on NHEFS:
- **Val RMSE: 20.64** (was 32.96)
- Still ranks ~30th/39 (worse than most models at ~6)
- But significantly better than before

### Why still worse than other models?

Even with beta=0.0, flow_ssc still has:
1. Complex 6-layer flow architecture for p(y|x,t)
2. Many parameters (~10k+) vs 50 training samples
3. Flow-based models need more data than discriminative models

**Models that do well on NHEFS** (RMSE 5.7-6.0):
- mean_teacher (simple consistency regularization)
- fixmatch (semi-supervised but simpler architecture)
- vat (virtual adversarial training, simpler)
- ss_cevae (VAE-based, has built-in regularization)

These models have:
- Fewer parameters
- Stronger regularization
- Better inductive biases for small data

## Conclusion

1. ✅ **Found root cause**: 50 samples × 65 dimensions breaks flow_x
2. ✅ **Found solution**: beta=0.0 improves by 37%
3. ⚠️ **Limitation**: flow_ssc still not competitive on tiny high-d datasets
4. 📝 **Recommendation**: Use adaptive beta OR document dataset requirements

The beta parameter successfully addresses the immediate issue, but flow_ssc should be used on datasets with n_samples > 200 and d_x < 50 for best performance.
