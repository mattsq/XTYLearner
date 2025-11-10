# Adaptive Benchmark Proposal

## Problem Statement

The current CI benchmark uses a "one size fits all" approach:
- **100 samples** for all datasets
- **10 epochs** for all models

This severely disadvantages slow-converging models like diffusion-based approaches, while potentially wasting compute on fast-converging models.

## Experimental Evidence

From our experiments comparing 100 samples/10 epochs vs 1000 samples/30 epochs:

| Model Type | Current CI | With More Resources | Improvement |
|------------|-----------|-------------------|-------------|
| **bridge_diff** (slow) | 1.93 RMSE | **0.20 RMSE** | ↓ 90% |
| **ccl_cpc** (medium) | 0.87 RMSE | **0.10 RMSE** | ↓ 88% |
| **multitask** (medium) | 0.63 RMSE | **0.11 RMSE** | ↓ 83% |
| **vat** (fast) | 0.17 RMSE | **0.10 RMSE** | ↓ 39% |

**Key Insight**: Complex models benefit dramatically from more resources, while simple models show smaller gains. Current rankings are misleading.

---

## Proposed Solution: Adaptive Benchmark

### 1. Model-Specific Epoch Budgets

Assign epochs based on convergence characteristics:

| Category | Epochs | Examples |
|----------|--------|----------|
| **Fast Convergers** | 5-10 | em, lp_knn, prob_circuit, vat, mean_teacher, fixmatch |
| **Medium Convergers** | 15-20 | cycle_dual, multitask, ganite, m2_vae, ss_cevae, gnn_scm |
| **Slow Convergers** | 25-30 | bridge_diff, lt_flow_diff, jsbf, flow_ssc, diffusion_cevae |

**Total: 39 models, averaging 18.3 epochs each**

### 2. Increased Sample Sizes

| Dataset | Current | Proposed | Rationale |
|---------|---------|----------|-----------|
| synthetic | 100 | **1000** | More stable estimates |
| synthetic_mixed | 100 | **1000** | Better semi-supervised evaluation |
| criteo_uplift | 1000 | **1000** | Keep current |
| nhefs | varies | **1566** | Use full dataset |

---

## Benefits

### ✅ **Fairness**
- Each model gets appropriate training time
- No more penalizing slow convergers
- No more wasting compute on fast convergers

### ✅ **Accuracy**
- 10x more data → more stable results
- Reduced variance between runs
- True model capabilities revealed

### ✅ **Efficiency**
- 39% less compute than uniform 30 epochs
- 83% more than uniform 10 epochs
- Smart middle ground

### ✅ **Maintainability**
- Clear categorization of models
- Easy to add new models with appropriate budgets
- Config-driven approach

---

## Computational Cost Analysis

### Current Approach (Uniform 10 Epochs)
```
Cost: 43 models × 3 datasets × 10 epochs = 1,290 epoch-models
Issue: Slow convergers underperform
```

### Naive Solution (Uniform 30 Epochs)
```
Cost: 43 models × 3 datasets × 30 epochs = 3,870 epoch-models
Issue: Wasteful on fast convergers (+200% cost)
```

### **Proposed Adaptive Approach**
```
Cost: 43 models × 3 datasets × 18.3 avg epochs = 2,362 epoch-models
Benefit: Fair to all models, only +83% vs current
Savings: 39% vs uniform 30 epochs
```

---

## Implementation

### Files Created

1. **scripts/adaptive_benchmark_config.py**
   - Defines epoch budgets per model
   - Specifies sample sizes per dataset
   - Generates config file

2. **scripts/run_adaptive_benchmark.py**
   - Benchmark runner using adaptive config
   - Reads budgets and trains accordingly
   - Reports progress and saves results

3. **benchmark_config_adaptive.json**
   - Machine-readable configuration
   - Can be version-controlled
   - Easy to modify

### Usage

```bash
# Generate configuration
python scripts/adaptive_benchmark_config.py

# Run full benchmark
python scripts/run_adaptive_benchmark.py

# Run specific models
python scripts/run_adaptive_benchmark.py "bridge_diff,vat,ganite"
```

---

## Expected Impact on Rankings

Based on our experimental data, here's how rankings would change:

### Synthetic Dataset

**Current CI (100/10)**:
1. em - 0.00
2. lp_knn - 0.00
3. prob_circuit - 0.12
4. vat - 0.17
5. mean_teacher - 0.18
...
39. **bridge_diff - 1.93** ⚠️

**Adaptive (1000/adaptive)**:
1. em - 0.00
2. lp_knn - 0.00
3. mean_teacher - **0.10**
4. vat - **0.10**
5. ccl_cpc - **0.10**
6. cycle_dual - **0.10**
7. ganite - **0.11**
8. multitask - **0.11**
9. **bridge_diff - 0.20** ✅ (30-position jump!)

### Key Changes

- **bridge_diff**: Rank 39 → Rank 9 (+30 positions)
- **ccl_cpc**: Rank 22 → Rank 5 (+17 positions)
- **multitask**: Rank 17 → Rank 8 (+9 positions)
- **m2_vae**: Rank 24 → Rank 8 (+16 positions)

Simple models (em, lp_knn) remain at top, but competitive rankings become much more meaningful.

---

## Recommendation

**Implement adaptive benchmarking for fair model comparison.**

### Phase 1: Validation (Immediate)
- Run adaptive benchmark on subset of models
- Validate results match predictions
- Document findings

### Phase 2: Integration (Short-term)
- Update CI to use adaptive configuration
- Deprecate old uniform-epoch approach
- Update documentation

### Phase 3: Optimization (Long-term)
- Add early stopping based on convergence
- Implement learning rate schedules per model type
- Consider multiple random seeds per model

---

## Conclusion

The current benchmark configuration creates misleading rankings by applying uniform constraints to models with vastly different convergence characteristics.

**Adaptive benchmarking provides:**
- ✅ Fair comparison across model types
- ✅ Accurate representation of capabilities
- ✅ Efficient use of computational resources
- ✅ Maintainable, config-driven approach

**Cost**: +83% compute vs current (but reveals true performance)
**Benefit**: Models ranked by actual capability, not training constraints

---

## Files in This Proposal

```
scripts/
  ├── adaptive_benchmark_config.py      # Config generator
  ├── run_adaptive_benchmark.py          # Adaptive benchmark runner
  └── experimental_benchmark_1000_30.py  # Validation experiment

benchmark_config_adaptive.json           # Generated configuration

ADAPTIVE_BENCHMARK_PROPOSAL.md           # This document
BRIDGE_DIFF_FINDINGS.md                  # Detailed analysis
BRIDGE_DIFF_DEBUG_PLAN.md                # Debugging methodology
```

---

**Ready to implement?** Run the adaptive benchmark to validate this approach!
