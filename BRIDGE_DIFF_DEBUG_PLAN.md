# Bridge_Diff Model Debugging and Testing Plan

## Executive Summary

This document outlines an exhaustive debugging and testing plan for the `bridge_diff` model, which is currently performing poorly (ranked 41/42 on synthetic dataset with val RMSE of 2.45512).

## Background

### Current Performance (from CI Baseline)
- **synthetic dataset**: val outcome rmse: 2.45512 (rank: 41/42)
- **synthetic_mixed dataset**: val outcome rmse: 1.67121 (rank: 75/84)

### Model Architecture
The BridgeDiff model (xtylearner/models/bridge_diff.py:70) consists of:
1. **ScoreBridge**: Score network predicting ∇_y log q(y_τ | x,t)
2. **Classifier**: Treatment classifier estimating p(t|x,y)
3. **Loss Components**:
   - Supervised loss (observed treatments)
   - Semi-supervised loss (unobserved treatments with posterior weighting)
   - Cross-entropy loss for treatment classification

### Recent History
- Commit a0e6b72 merged optimizations to improve runtime and training signal
- Still showing poor performance despite optimizations

## Phase 1: Code Review and Static Analysis

### 1.1 Mathematical Correctness Review
- [ ] Verify score network loss formulation (line 140: `(s_obs + eps[obs_mask] * inv_sig) ** 2`)
- [ ] Check sigma scheduling (lines 99-106: logarithmic interpolation)
- [ ] Validate semi-supervised loss weighting (line 166: posterior probability weighting)
- [ ] Review noise scale computation and clamping logic
- [ ] Verify gradient flow through all loss components

### 1.2 Numerical Stability Analysis
- [ ] Check for potential division by zero (eps_stab = 1e-6 usage)
- [ ] Verify sigma clamping (sigma_min=0.002, sigma_max=1.0)
- [ ] Review loss scaling (sig_obs**2 * mse weighting)
- [ ] Check for NaN/Inf propagation points
- [ ] Validate tau dimension handling (lines 44-47)

### 1.3 Sampling Implementation Review
- [ ] Review paired_sample method (lines 181-229)
- [ ] Verify reverse diffusion steps
- [ ] Check noise injection schedule (lines 218-223)
- [ ] Validate sigma transition between steps
- [ ] Review treatment indexing in multi-treatment sampling

### 1.4 Hyperparameter Analysis
- [ ] Default timesteps=1000 vs actual sampling steps (min(timesteps, 50))
- [ ] sigma_min/sigma_max range appropriateness
- [ ] hidden=256, embed_dim=64, n_blocks=3 adequacy
- [ ] Learning rate defaults

## Phase 2: Unit Testing Suite

### 2.1 Component-Level Tests
Create `tests/test_bridge_diff_detailed.py` with:

#### Test ScoreBridge Network
```python
def test_score_bridge_forward():
    # Test forward pass shapes
    # Test with different tau dimensions
    # Test embedding dimensions
    # Test gradient flow
```

#### Test Classifier
```python
def test_classifier_probabilities():
    # Test probability sums to 1
    # Test logits shape
    # Test gradient flow
```

#### Test Sigma Schedule
```python
def test_sigma_schedule():
    # Test monotonic increase
    # Test boundary values
    # Test clamping behavior
    # Test gradient computation
```

### 2.2 Loss Function Tests
```python
def test_loss_observed_only():
    # Test with only observed treatments (t_obs != -1)
    # Verify loss computation
    # Check gradient norms

def test_loss_unobserved_only():
    # Test with only unobserved treatments (t_obs == -1)
    # Verify posterior weighting
    # Check loss components

def test_loss_mixed():
    # Test with mixed observed/unobserved
    # Verify both components active
    # Check relative magnitudes

def test_loss_warmup():
    # Test warmup behavior
    # Verify unobserved loss disabled during warmup
```

### 2.3 Sampling Tests
```python
def test_paired_sample_shapes():
    # Test output shapes for different n_samples
    # Test return_tensor behavior
    # Test tuple unpacking

def test_paired_sample_convergence():
    # Test that samples converge from noise
    # Check sigma reduction over steps
    # Verify final samples are reasonable

def test_predict_outcome_consistency():
    # Test prediction consistency across calls
    # Test n_samples averaging
    # Verify treatment indexing
```

## Phase 3: Integration and Behavior Tests

### 3.1 Training Behavior Analysis
```python
def test_bridge_diff_training_convergence():
    # Train on simple synthetic data
    # Monitor loss components over epochs
    # Check for divergence/explosion
    # Verify gradient norms stay bounded

def test_bridge_diff_overfitting_check():
    # Train on tiny dataset
    # Verify can achieve low training loss
    # Check if model has capacity to fit data
```

### 3.2 Comparison Against Baselines
```python
def test_bridge_diff_vs_simple_baseline():
    # Compare against mean prediction
    # Compare against linear regression
    # Should outperform on train set at minimum
```

### 3.3 Data Dependency Tests
```python
def test_bridge_diff_label_ratio_sensitivity():
    # Test with label_ratio from 0.1 to 0.9
    # Check performance degradation pattern
    # Verify semi-supervised component helps

def test_bridge_diff_treatment_balance():
    # Test with balanced/imbalanced treatments
    # Check classifier performance
    # Verify posterior probabilities
```

## Phase 4: Diagnostic Runs with Baseline Comparison

### 4.1 Reproduce CI Baseline
```bash
# Run exact same setup as CI
python eval.py --model bridge_diff --dataset synthetic --output bridge_diff_synthetic_debug.json
python eval.py --model bridge_diff --dataset synthetic_mixed --output bridge_diff_mixed_debug.json
```

Expected Results (to match CI baseline):
- synthetic: val_outcome_rmse ≈ 2.45512
- synthetic_mixed: val_outcome_rmse ≈ 1.67121

### 4.2 Loss Component Analysis
Add instrumentation to track:
- loss_obs magnitude over epochs
- loss_unobs magnitude over epochs
- ce_loss magnitude over epochs
- Ratio between components
- Gradient norms for each component

### 4.3 Compare with Similar Models
Compare bridge_diff against:
- `lt_flow_diff` (similar diffusion-based approach)
- `diffusion_cevae` (diffusion-based generative model)
- `flow_ssc` (flow-based semi-supervised)

## Phase 5: Potential Issues to Investigate

### 5.1 Known Potential Issues

1. **Loss Weighting Imbalance**
   - Line 166: Semi-supervised loss weighted by p_post * weight * mse
   - Might be too weak/strong relative to supervised loss
   - **Test**: Add logging to compare loss_obs vs loss_unobs magnitude

2. **Sigma Schedule Too Aggressive**
   - Default sigma_max=1.0 might be too large for outcome scale
   - Noise might dominate signal
   - **Test**: Try sigma_max=0.1, 0.5 and compare

3. **Insufficient Sampling Steps**
   - Line 194: n_steps = min(timesteps, 50)
   - Default only 50 steps despite timesteps=1000
   - **Test**: Increase n_steps to 100, 200

4. **Score Network Capacity**
   - embed_dim=64, hidden=256, n_blocks=3
   - Might be insufficient for complex distributions
   - **Test**: Double hidden size to 512

5. **Learning Rate Issues**
   - No explicit default_lr attribute
   - Falls back to 0.001 in benchmark
   - **Test**: Try 0.0001, 0.0005, 0.001, 0.005

6. **Tau Embedding Issues**
   - Lines 44-47: Dimension handling for tau
   - Potential shape mismatch
   - **Test**: Verify tau is always correctly shaped

7. **Score Target Mismatch**
   - Line 140: Training target is -eps / sigma (standardized noise)
   - Might not match true score of bridge process
   - **Test**: Review diffusion bridge theory

8. **Posterior Probability Quality**
   - Classifier might produce poor posteriors early in training
   - Semi-supervised loss might be misleading
   - **Test**: Monitor classifier accuracy during training

9. **Warmup Not Used**
   - warmup parameter defaults to 0
   - Semi-supervised component active from start
   - **Test**: Try warmup=5 epochs

10. **EPS Clipping Too Tight**
    - eps_stab = 1e-6 might be too small
    - Could cause numerical issues
    - **Test**: Try 1e-4, 1e-5

### 5.2 Comparison with Better-Performing Models

Top performers on synthetic dataset for reference:
1. em: 0.0 (but might be trivial)
2. lp_knn: 0.0
3. prob_circuit: 0.11587
4. cycle_dual: 0.146573

Bridge_diff should at least beat simple baselines.

## Phase 6: Systematic Debugging Protocol

### Step-by-step debugging approach:

1. **Sanity Check** (30 min)
   - Run existing test: `pytest tests/test_trainer.py::test_bridge_diff_trainer_runs -v`
   - Verify test passes
   - Check if loss decreases over epochs

2. **Component Tests** (2 hours)
   - Implement and run all unit tests from Phase 2
   - Identify any failing components
   - Fix fundamental issues

3. **Loss Analysis** (1 hour)
   - Add detailed logging to loss() method
   - Run 10 epoch training
   - Analyze loss component evolution

4. **Baseline Reproduction** (1 hour)
   - Run eval.py on both datasets
   - Verify results match CI baseline
   - Document any differences

5. **Hyperparameter Sweep** (2 hours)
   - Test sigma_max: [0.1, 0.5, 1.0]
   - Test n_steps: [50, 100, 200]
   - Test hidden: [256, 512]
   - Test lr: [0.0001, 0.001, 0.01]
   - Test warmup: [0, 5, 10]

6. **Compare with Similar Models** (1 hour)
   - Run lt_flow_diff and diffusion_cevae
   - Compare architectures
   - Identify key differences

7. **Root Cause Identification** (1 hour)
   - Synthesize findings
   - Identify top 3 likely issues
   - Propose fixes

8. **Fix Implementation** (2 hours)
   - Implement fixes
   - Run validation tests
   - Compare with baseline

9. **Final Validation** (1 hour)
   - Run full benchmark suite
   - Verify improvement
   - Document changes

## Phase 7: Deliverables

1. **Test Suite**
   - `tests/test_bridge_diff_detailed.py` with comprehensive unit tests
   - Tests for all components and integration points

2. **Diagnostic Report**
   - `BRIDGE_DIFF_DIAGNOSIS.md` with findings
   - Loss component analysis
   - Hyperparameter sensitivity results

3. **Performance Comparison**
   - Before/after metrics
   - Comparison with CI baseline
   - Comparison with similar models

4. **Code Fixes**
   - Implementation of identified fixes
   - Comments explaining changes
   - Updated docstrings

5. **Updated Benchmark**
   - New benchmark results
   - Performance improvement quantification

## Success Criteria

1. **Minimum**: Model doesn't fail/crash, produces finite loss
2. **Expected**: Val RMSE < 1.5 on synthetic (better than current 2.45)
3. **Target**: Val RMSE < 1.0 on synthetic (competitive with top 20)
4. **Stretch**: Val RMSE < 0.5 on synthetic (competitive with top 10)

## Timeline Estimate

- Phase 1: 2 hours (code review)
- Phase 2: 3 hours (unit tests)
- Phase 3: 2 hours (integration tests)
- Phase 4: 2 hours (diagnostic runs)
- Phase 5: 1 hour (issue investigation)
- Phase 6: 10 hours (systematic debugging)
- Phase 7: 2 hours (documentation)

**Total: ~22 hours**

## Next Steps

1. Begin with Phase 1: Code review
2. Create test suite infrastructure
3. Run baseline reproduction
4. Execute systematic debugging protocol
5. Document and fix identified issues
