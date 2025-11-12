# Proposed New Models for XTYLearner

*Generated: 2025-11-12*
*Current model count: 41*

This document proposes new models to extend XTYLearner's capabilities based on recent research and identified gaps in the current model catalog.

---

## 1. Transformer-Based Models

### 1.1 DAG-Aware Transformer (`dag_transformer`)
**Category**: Discriminative
**Priority**: HIGH
**Reference**: arXiv:2410.10044 (October 2024)

**Description**: A transformer that explicitly incorporates causal DAG structure into the attention mechanism, enabling it to model causal relationships while estimating treatment effects.

**Key Features**:
- DAG-aware attention mechanism that respects causal ordering
- Flexible estimation of both ATE and CATE
- Outperforms Generalized Random Forests in CATE estimation
- Can handle semi-supervised settings with missing treatment labels

**Implementation Notes**:
- Extend `MaskedTabularTransformer` with DAG-aware attention
- Add learnable or input-specified DAG structure
- Support both known and learned causal graphs

---

### 1.2 Causal Prior-Data Fitted Network (`causal_pfn`)
**Category**: Discriminative/Foundation Model
**Priority**: HIGH
**Reference**: CausalFM framework (2024)

**Description**: A transformer-based foundation model pre-trained on synthetic causal data that performs in-context learning for treatment effect estimation.

**Key Features**:
- Pre-trained on diverse synthetic causal scenarios
- Enables few-shot causal inference through in-context learning
- No fine-tuning required for new datasets
- Competitive CATE estimation on small datasets

**Implementation Notes**:
- Implement transformer architecture with causal inductive biases
- Create synthetic data generation pipeline for pre-training
- Support in-context learning interface
- May require substantial compute for pre-training

---

### 1.3 Causal Transformer (`causal_transformer`)
**Category**: Discriminative
**Priority**: MEDIUM
**Reference**: ICML 2022 (Melnychuk et al.)

**Description**: Transformer architecture for counterfactual estimation using sequence-to-sequence modeling of potential outcomes.

**Key Features**:
- Treats counterfactual inference as sequence prediction
- Self-attention over treatment and covariate sequences
- Handles time-varying treatments naturally
- Position encodings for treatment timing

**Implementation Notes**:
- Encoder-decoder transformer architecture
- Specialized for sequential treatment settings
- Can be adapted for static treatments as well

---

## 2. Meta-Learner Architectures

### 2.1 Neural T-Learner (`t_learner`)
**Category**: Discriminative
**Priority**: HIGH
**Reference**: PNAS 2019 (Künzel et al.)

**Description**: Two separate neural networks for control and treatment groups, with CATE estimated as the difference in predictions.

**Key Features**:
- Simple and interpretable approach
- Separate modeling prevents treatment/control interference
- Works with any base neural architecture
- Semi-supervised extension possible

**Implementation Notes**:
- Two independent neural networks
- Optional weight sharing in early layers
- Support for continuous treatments via multiple networks

---

### 2.2 Neural X-Learner (`x_learner`)
**Category**: Discriminative
**Priority**: HIGH
**Reference**: PNAS 2019 (Künzel et al.)

**Description**: Three-stage meta-learner that models individual treatment effects in a cross-fitting procedure, particularly effective when treatment groups are imbalanced.

**Key Features**:
- Stage 1: Fit outcome models for each treatment
- Stage 2: Impute treatment effects and model them
- Stage 3: Combine estimates using propensity weights
- Handles imbalanced treatment assignment well

**Implementation Notes**:
- Multi-stage training procedure
- Requires propensity score estimation
- Can leverage existing outcome networks from the codebase

---

### 2.3 Neural R-Learner (`r_learner`)
**Category**: Discriminative
**Priority**: MEDIUM
**Reference**: PNAS 2019 + Robinson decomposition

**Description**: Directly models the CATE function using residualized outcomes and treatments, leveraging Robinson's double-residual approach.

**Key Features**:
- Orthogonal estimation of treatment effects
- Reduces bias from confounding
- Theoretically grounded via partial identification
- Works with high-dimensional confounders

**Implementation Notes**:
- Two-stage: (1) residualize Y and T, (2) regress residuals
- Can use any neural architecture for both stages
- Semi-supervised variants possible

---

### 2.4 Doubly Robust Learner (`dr_learner`)
**Category**: Discriminative
**Priority**: MEDIUM
**Reference**: Augmented IPW literature

**Description**: Combines outcome regression and propensity score weighting for doubly robust treatment effect estimation.

**Key Features**:
- Consistent if either propensity or outcome model is correct
- Lower variance than pure IPW methods
- Semi-supervised extension via pseudo-labeling
- Handles missing treatment labels naturally

**Implementation Notes**:
- Joint training of propensity and outcome networks
- AIPW (Augmented IPW) loss function
- Can build on existing `DragonNet` architecture

---

## 3. Advanced Graph-Based Methods

### 3.1 Heterogeneous-Effect Causal Graph Diffusion (`he_cgdn`)
**Category**: Diffusion + Graph
**Priority**: MEDIUM
**Reference**: Springer 2025

**Description**: Combines causal graph learning, heterogeneous treatment effect estimation, and graph diffusion for interpretable causal inference.

**Key Features**:
- Learns causal graph structure from data
- Estimates heterogeneous effects via graph neural networks
- Diffusion process for effect propagation
- Highly interpretable causal mechanisms

**Implementation Notes**:
- Extend `diffusion_gnn_scm` with heterogeneous effects
- Add causal discovery module (e.g., NOTEARS)
- Implement graph diffusion for treatment effect propagation

---

### 3.2 Causal Graph Attention Network (`cgate`)
**Category**: Discriminative + Graph
**Priority**: MEDIUM

**Description**: Graph attention network specialized for causal inference that learns attention weights based on causal relationships.

**Key Features**:
- Attention mechanism prioritizes causal parents
- Handles interference and spillover effects
- Can incorporate domain knowledge via attention priors
- Semi-supervised via graph regularization

**Implementation Notes**:
- GAT architecture with causal attention constraints
- Support for known/unknown graph structures
- Can handle networked observational data

---

## 4. Causal Representation Learning

### 4.1 Instrumental Variable Network (`iv_net`)
**Category**: Discriminative
**Priority**: HIGH
**Reference**: Deep IV (ICML 2017) + recent extensions

**Description**: Neural network approach to instrumental variable estimation for handling unmeasured confounding.

**Key Features**:
- Two-stage least squares via neural networks
- Handles unmeasured confounding with valid instruments
- Non-linear treatment and outcome relationships
- Semi-supervised possible with unlabeled outcomes

**Implementation Notes**:
- Two-stage architecture: (1) IV → Treatment, (2) Treatment → Outcome
- Adversarial training for stage 1
- Can handle weak instruments via regularization

---

### 4.2 Disentangled Causal Representation (`dcr`)
**Category**: Generative
**Priority**: MEDIUM
**Reference**: CausalVAE + disentanglement literature

**Description**: VAE that learns disentangled representations separating confounders, instrumental variables, and adjustment variables.

**Key Features**:
- Factorized latent space with causal structure
- Identifies which latents affect T, Y, or both
- Enables causal reasoning in latent space
- Semi-supervised via missing treatment labels

**Implementation Notes**:
- Extend existing VAE models with disentanglement losses
- Add causal structure constraints (e.g., β-VAE + causal)
- Factor-style decomposition of latents

---

### 4.3 Proximal Causal Learner (`proximal_learner`)
**Category**: Discriminative
**Priority**: MEDIUM
**Reference**: Proximal causal inference literature (2022+)

**Description**: Handles unmeasured confounding using proxy variables for unobserved confounders.

**Key Features**:
- Uses treatment/outcome confounding bridges (proxies)
- Identifies causal effects without full confounding control
- Neural implementation of proximal g-formula
- Particularly useful for real-world observational data

**Implementation Notes**:
- Requires specification of treatment and outcome proxies
- Two-bridge architecture: Z → U ← W
- Non-parametric via neural networks

---

## 5. Advanced Generative Models

### 5.1 Flow Matching for Causal Inference (`flow_matching`)
**Category**: Generative
**Priority**: MEDIUM
**Reference**: Flow Matching (2022) + causal applications

**Description**: Continuous normalizing flow using optimal transport for modeling counterfactual distributions.

**Key Features**:
- Simulation-free training (unlike diffusion models)
- Faster sampling than diffusion
- Direct transport between factual and counterfactual
- Handles missing treatments naturally

**Implementation Notes**:
- Vector field network for conditional flow
- Optimal transport loss for training
- Can build on existing `cnflow_model` infrastructure

---

### 5.2 Rectified Flow for Counterfactuals (`rectified_flow`)
**Category**: Generative
**Priority**: LOW
**Reference**: Rectified Flow (NeurIPS 2022)

**Description**: Learns straight-line trajectories between factual and counterfactual outcomes for fast generation.

**Key Features**:
- Straighter paths than standard flows/diffusion
- Fewer sampling steps required
- Distillation-friendly for deployment
- Handles treatment as conditioning variable

**Implementation Notes**:
- Rectification procedure for flow straightening
- Can start from existing diffusion models
- Iterative refinement process

---

### 5.3 Energy-Based CATE Model (`ebm_cate`)
**Category**: Generative
**Priority**: MEDIUM

**Description**: Energy-based model that directly learns the CATE function via contrastive estimation.

**Key Features**:
- Flexible density estimation for CATE distribution
- Uncertainty quantification via energy landscape
- Handles complex, multimodal treatment effects
- Semi-supervised via unlabeled data contrastive loss

**Implementation Notes**:
- Extend `joint_ebm` to focus on conditional effects
- Langevin sampling for inference
- Noise contrastive estimation for training

---

## 6. Bayesian Neural Approaches

### 6.1 Bayesian Causal Forest Network (`bcf_net`)
**Category**: Discriminative
**Priority**: HIGH
**Reference**: Neural BCF implementations (2024)

**Description**: Neural network implementation of Bayesian Causal Forests with uncertainty quantification.

**Key Features**:
- Bayesian treatment effect estimation
- Credible intervals for CATE
- Regularization via outcome-focused priors
- Handles small treatment effects well

**Implementation Notes**:
- Variational inference for posterior approximation
- Monte Carlo dropout or ensemble approaches
- Prior regularization for treatment vs. prognostic effects

---

### 6.2 Neural BART (`neural_bart`)
**Category**: Discriminative
**Priority**: MEDIUM
**Reference**: BART + neural network literature

**Description**: Bayesian Additive Regression Trees implemented via neural networks with additive structure.

**Key Features**:
- Additive neural network structure
- Bayesian regularization and sum-of-trees prior
- Automatic relevance determination
- Works well with weak effects

**Implementation Notes**:
- Multiple shallow networks (tree surrogates)
- Additive aggregation with learned weights
- Variational inference for training

---

### 6.3 Gaussian Process Treatment Effects (`gp_te`)
**Category**: Discriminative
**Priority**: LOW
**Reference**: GP causal inference literature

**Description**: Gaussian Process models for treatment effect functions with principled uncertainty.

**Key Features**:
- Non-parametric treatment effect functions
- Exact uncertainty quantification
- Works well with small/medium data
- Can incorporate causal structure via kernel design

**Implementation Notes**:
- Sparse GP approximations for scalability
- Treatment-specific kernels
- Mean function from neural network (Deep Kernel Learning)

---

## 7. Multi-Treatment Extensions

### 7.1 Dynamic Neural Masking (`dynamic_mask`)
**Category**: Discriminative
**Priority**: MEDIUM
**Reference**: arXiv:2511.01641 (November 2024)

**Description**: Handles multiple categorical treatments with different cardinalities via dynamic masking.

**Key Features**:
- Cross-treatment effect estimation
- Handles treatments with varying numbers of levels
- Learns treatment embeddings jointly
- Semi-supervised across treatment types

**Implementation Notes**:
- Masked prediction heads for each treatment type
- Shared encoder with treatment-specific decoders
- Can extend `masked_tabular_transformer`

---

### 7.2 Dose-Response Network (`dose_net`)
**Category**: Discriminative
**Priority**: MEDIUM
**Reference**: Dose-response curve literature + GPS

**Description**: Specialized network for continuous treatment doses with generalized propensity score.

**Key Features**:
- Models full dose-response curve
- Generalized propensity score for continuous T
- Shape constraints (monotonicity, convexity)
- Optimal dose recommendation

**Implementation Notes**:
- Smooth network with dose as input
- GPS estimation network
- Shape-constrained architectures (monotonic networks)

---

## 8. Domain Adaptation & Transfer

### 8.1 Causal Domain Adaptation (`causal_da`)
**Category**: Discriminative
**Priority**: MEDIUM

**Description**: Transfers causal models across domains with distribution shift.

**Key Features**:
- Learns domain-invariant causal representations
- Adapts treatment effects across populations
- Handles covariate shift and target shift
- Semi-supervised domain adaptation

**Implementation Notes**:
- Adversarial domain discriminator
- Causal invariance constraints
- Can build on existing discriminative models

---

### 8.2 Meta-Learning for Causal Inference (`maml_causal`)
**Category**: Discriminative
**Priority**: LOW
**Reference**: MAML + causal inference

**Description**: Meta-learning approach for quick adaptation to new causal inference tasks.

**Key Features**:
- Few-shot causal effect estimation
- Rapid adaptation to new treatment/outcome pairs
- Learns generalizable causal structure
- Useful for multiple related studies

**Implementation Notes**:
- MAML-style meta-training loop
- Task = new treatment effect estimation problem
- Inner loop adapts to specific task

---

## 9. Specialized Architectures

### 9.1 Causal Convolutional Network (`causal_cnn`)
**Category**: Discriminative
**Priority**: LOW

**Description**: 1D convolutional network with causal (forward-only) convolutions for sequential treatments.

**Key Features**:
- Respects temporal causal order
- Efficient for long sequences
- Handles time-varying confounding
- Can model treatment history effects

**Implementation Notes**:
- Causal padding (no future info)
- Temporal receptive field growth
- Useful for longitudinal data

---

### 9.2 Neural Structural Model (`nsm`)
**Category**: Generative
**Priority**: MEDIUM
**Reference**: Structural causal models + neural networks

**Description**: Explicitly models the structural causal model with neural networks for each mechanism.

**Key Features**:
- Separate networks for each causal mechanism
- Identifiable under certain conditions
- Interpretable causal structure
- Enables counterfactual reasoning

**Implementation Notes**:
- One network per variable in causal graph
- Structural equation modeling via NNs
- Can handle cycles via equilibrium finding

---

### 9.3 Copula-Based Neural Model (`copula_net`)
**Category**: Generative
**Priority**: LOW
**Reference**: Copula-based causal inference (MDPI 2024)

**Description**: Models joint distribution of (X,T,Y) using copulas with neural network marginals.

**Key Features**:
- Flexible dependence modeling via copulas
- Neural marginal distributions
- Handles survival outcomes naturally
- Can model complex dependence structures

**Implementation Notes**:
- Neural networks for marginals
- Copula family selection (Gaussian, Archimedean, etc.)
- Vine copulas for high dimensions

---

## 10. Efficiency and Scalability

### 10.1 Quantized Treatment Effect Model (`qte_model`)
**Category**: Discriminative
**Priority**: LOW

**Description**: Efficient model using quantization for deployment in resource-constrained settings.

**Key Features**:
- 8-bit or lower weight quantization
- Maintains treatment effect accuracy
- Fast inference for production systems
- Can quantize any existing model

**Implementation Notes**:
- Post-training quantization
- Quantization-aware training
- Distillation from full-precision models

---

### 10.2 Mixture of Experts CATE (`moe_cate`)
**Category**: Discriminative
**Priority**: MEDIUM

**Description**: Mixture of experts where each expert specializes in different treatment effect regimes.

**Key Features**:
- Automatic discovery of subpopulations
- Each expert learns effects for different groups
- Gating network routes samples to experts
- Interpretable via expert assignment

**Implementation Notes**:
- Multiple small networks (experts)
- Gating network based on X
- Can use sparsity to activate few experts

---

## Summary & Recommendations

### Highest Priority (Implement First)
1. **Neural T-Learner** - Simple, effective baseline
2. **Neural X-Learner** - Strong for imbalanced treatments
3. **DAG-Aware Transformer** - Cutting-edge architecture
4. **Causal PFN** - Foundation model approach
5. **Bayesian Causal Forest Network** - Uncertainty quantification
6. **IV Network** - Handles unmeasured confounding

### Medium Priority (Next Phase)
- Neural R-Learner and DR-Learner (complete meta-learner suite)
- Flow Matching (faster alternative to diffusion)
- HE-CGDN (graph + diffusion + causal)
- Dose-Response Network (continuous treatments)
- Proximal Causal Learner (proxy variables)

### Lower Priority (Future Work)
- Causal CNN (sequential treatments)
- Meta-learning approaches
- Quantized models (deployment)
- Copula-based models (specialized use cases)

---

## Implementation Strategy

### Phase 1: Meta-Learners (Foundation)
Add T-learner, X-learner, R-learner, DR-learner as they provide strong baselines and are well-understood.

### Phase 2: Advanced Architectures (Innovation)
Implement DAG-aware transformer and Causal PFN for state-of-the-art performance.

### Phase 3: Specialized Methods (Coverage)
Add IV networks, proximal learners, and dose-response networks for specific problem settings.

### Phase 4: Efficiency & Scale (Deployment)
Implement quantization, distillation, and production-ready variants.

---

## References

Key papers for implementation:
- Künzel et al. (2019): "Metalearners for estimating heterogeneous treatment effects"
- Shalit et al. (2017): "Estimating individual treatment effect: generalization bounds and algorithms" (CFRNet)
- Hartford et al. (2017): "Deep IV: A flexible approach for counterfactual prediction"
- Melnychuk et al. (2022): "Causal Transformer for Estimating Counterfactual Outcomes"
- DAG-aware Transformer (2024): arXiv:2410.10044
- CausalFM (2024): Prior-Data Fitted Networks for Causal Inference
- Hassanpour & Greiner (2020): "Learning Disentangled Representations for CounterFactual Regression"
- Shi et al. (2019): "Adapting Neural Networks for the Estimation of Treatment Effects"

