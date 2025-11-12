# Proposed Models - Executive Summary

## Overview
This document proposes **22 new models** to complement the existing 41 models in XTYLearner, based on recent research (2024-2025) and identified gaps.

---

## Quick Reference Table

| # | Model Name | Category | Priority | Key Innovation |
|---|------------|----------|----------|----------------|
| 1 | DAG-Aware Transformer | Discriminative | HIGH | Incorporates causal graph structure into attention |
| 2 | Causal PFN | Foundation | HIGH | Pre-trained transformer for few-shot causal inference |
| 3 | Causal Transformer | Discriminative | MEDIUM | Sequence-to-sequence counterfactual modeling |
| 4 | Neural T-Learner | Meta-learner | HIGH | Separate networks for treatment/control |
| 5 | Neural X-Learner | Meta-learner | HIGH | Three-stage cross-fitting for imbalanced data |
| 6 | Neural R-Learner | Meta-learner | MEDIUM | Double-residual orthogonal estimation |
| 7 | DR-Learner | Meta-learner | MEDIUM | Doubly robust augmented IPW |
| 8 | HE-CGDN | Graph + Diffusion | MEDIUM | Graph diffusion with heterogeneous effects |
| 9 | Causal GAT | Graph | MEDIUM | Graph attention for causal relationships |
| 10 | IV Network | Discriminative | HIGH | Instrumental variables for unmeasured confounding |
| 11 | Disentangled Causal Rep | Generative | MEDIUM | VAE with factorized causal latents |
| 12 | Proximal Learner | Discriminative | MEDIUM | Uses proxy variables for confounding |
| 13 | Flow Matching | Generative | MEDIUM | Faster alternative to diffusion models |
| 14 | Rectified Flow | Generative | LOW | Straight-line counterfactual trajectories |
| 15 | EBM-CATE | Generative | MEDIUM | Energy-based CATE estimation |
| 16 | Bayesian CF Network | Discriminative | HIGH | Neural BCF with uncertainty quantification |
| 17 | Neural BART | Discriminative | MEDIUM | Additive neural trees |
| 18 | GP Treatment Effects | Discriminative | LOW | Gaussian process CATE |
| 19 | Dynamic Masking | Multi-treatment | MEDIUM | Multi-category treatment effects |
| 20 | Dose-Response Net | Discriminative | MEDIUM | Continuous dose curves with GPS |
| 21 | Causal Domain Adapt | Discriminative | MEDIUM | Transfer effects across populations |
| 22 | Mixture of Experts CATE | Discriminative | MEDIUM | Subpopulation-specific experts |

---

## Top 6 Recommendations (Implement First)

### 1. Neural T-Learner & X-Learner
**Why**: Industry-standard meta-learners, proven effectiveness, simple to implement
**Impact**: Provides strong baselines for all benchmarks
**Effort**: Low (can reuse existing network architectures)

### 2. DAG-Aware Transformer
**Why**: State-of-the-art (Oct 2024), incorporates causal structure explicitly
**Impact**: Outperforms GRF on CATE estimation, handles complex dependencies
**Effort**: Medium (extend existing transformer architecture)

### 3. Causal Prior-Data Fitted Network (PFN)
**Why**: Foundation model approach, few-shot learning capability
**Impact**: Strong performance on small datasets without fine-tuning
**Effort**: High (requires pre-training infrastructure)

### 4. Instrumental Variable Network
**Why**: Only approach that handles unmeasured confounding
**Impact**: Enables causal inference in challenging observational settings
**Effort**: Medium (two-stage architecture)

### 5. Bayesian Causal Forest Network
**Why**: Uncertainty quantification, handles weak treatment effects
**Impact**: Provides credible intervals, critical for decision-making
**Effort**: Medium (variational inference)

### 6. Doubly Robust Learner
**Why**: Robustness to model misspecification
**Impact**: Consistent under weaker assumptions than outcome modeling alone
**Effort**: Low (extend DragonNet architecture)

---

## Gap Analysis

### Current Strengths
✅ Diffusion models (7 models)
✅ Semi-supervised baselines (6 models)
✅ Generative models (13 models)
✅ Novel architectures (JEPA, CPC, masking)

### Identified Gaps
❌ **No meta-learners** (T, X, R, S, DR-learners)
❌ **Limited transformer models** (only 2 of 41)
❌ **No foundation models** (no pre-training approaches)
❌ **No IV methods** (can't handle unmeasured confounding)
❌ **Limited Bayesian approaches** (no uncertainty quantification)
❌ **No domain adaptation** (can't transfer across populations)
❌ **Limited multi-treatment** (only basic support)

---

## Implementation Phases

### Phase 1: Fill Critical Gaps (3-4 months)
- [ ] T-Learner, X-Learner (2 weeks)
- [ ] R-Learner, DR-Learner (2 weeks)
- [ ] IV Network (3 weeks)
- [ ] Bayesian CF Network (3 weeks)

**Deliverable**: Complete meta-learner suite + unmeasured confounding + UQ

### Phase 2: Advanced Architectures (3-4 months)
- [ ] DAG-Aware Transformer (4 weeks)
- [ ] Causal PFN (6 weeks - includes pre-training)
- [ ] Flow Matching (3 weeks)
- [ ] Proximal Learner (3 weeks)

**Deliverable**: State-of-the-art methods + foundation model

### Phase 3: Specialized Methods (2-3 months)
- [ ] Dose-Response Network (2 weeks)
- [ ] Dynamic Neural Masking (3 weeks)
- [ ] HE-CGDN (4 weeks)
- [ ] Causal Domain Adaptation (3 weeks)

**Deliverable**: Expanded coverage of problem settings

### Phase 4: Production & Scale (2 months)
- [ ] Model quantization (2 weeks)
- [ ] Distillation pipelines (2 weeks)
- [ ] MoE-CATE (3 weeks)
- [ ] Deployment optimization (2 weeks)

**Deliverable**: Production-ready, efficient models

---

## Expected Impact

### Research Impact
- **Meta-learners**: Direct comparison with classical methods
- **DAG-aware models**: Leverage causal knowledge
- **Foundation models**: Transfer learning for causal inference
- **IV methods**: Expand applicability to confounded settings

### Practical Impact
- **Uncertainty quantification**: Better decision-making
- **Smaller data**: Foundation models work with limited samples
- **Robustness**: Doubly robust methods reduce misspecification risk
- **Interpretability**: Meta-learners and DAG models are more transparent

### Benchmark Performance
Expected improvements on existing benchmarks:
- IHDP: +5-10% PEHE reduction (meta-learners, Bayesian methods)
- Jobs: +10-15% ATE accuracy (DR-learner, IV network)
- TWINS: +5-8% PEHE reduction (X-learner, DAG-aware)

---

## Resource Requirements

### Computational
- **Meta-learners**: Low (same as existing models)
- **Transformers**: Medium-High (attention scales quadratically)
- **PFN Pre-training**: Very High (requires large-scale synthetic data)
- **Bayesian methods**: Medium (variational inference)

### Implementation Time
- **Quick wins** (2-4 weeks each): T/X/R/DR-learners, DR-learner extension
- **Medium effort** (4-6 weeks each): DAG transformer, IV network, Bayesian CF
- **Major projects** (8+ weeks): Causal PFN, HE-CGDN

### Maintenance
- Meta-learners: Low (stable algorithms)
- Advanced architectures: Medium (evolving research)
- Foundation models: High (requires retraining as data grows)

---

## Selection Criteria

When prioritizing which models to implement:

1. **Scientific rigor**: Well-established theoretical foundations
2. **Empirical validation**: Strong benchmark performance in literature
3. **Complementarity**: Fills gap in existing model catalog
4. **Practical utility**: Solves real-world problems
5. **Implementation feasibility**: Reasonable effort to implement
6. **Maintenance burden**: Sustainable to maintain long-term

---

## Next Steps

1. **Review proposals** with research team
2. **Prioritize models** based on project goals
3. **Create implementation plan** for Phase 1
4. **Set up benchmarking** for new models
5. **Establish evaluation metrics** beyond PEHE/ATE
6. **Plan documentation** and tutorials

---

## References

See `PROPOSED_MODELS.md` for detailed descriptions, implementation notes, and full references for each model.

**Key Survey Papers**:
- Koch et al. (2025): "A Primer on Deep Learning for Causal Inference"
- arXiv:2405.03130: "Deep Learning for Causal Inference: A Comparison of Architectures"
- Künzel et al. (2019): "Metalearners for estimating heterogeneous treatment effects"

**Recent Innovations**:
- arXiv:2410.10044: DAG-aware Transformer (Oct 2024)
- CausalFM (2024): Prior-Data Fitted Networks
- arXiv:2511.01641: Dynamic Neural Masking (Nov 2024)
