# XTYLearner Model Selection Guide

This guide helps you choose the right model for your causal inference and treatment effect estimation tasks. XTYLearner provides 40+ models spanning classical semi-supervised methods to cutting-edge diffusion and generative approaches.

## Quick Reference

| Goal | Recommended Models |
|------|-------------------|
| Fast baseline | `lp_knn`, `multitask`, `mean_teacher` |
| Robust causal effects | `dragon_net`, `cacore`, `deconfounder_cfm` |
| Missing treatment labels | `jsbf`, `bridge_diff`, `ss_cevae`, `gnn_scm` |
| Learn causal structure | `gnn_scm`, `diffusion_gnn_scm`, `dag_transformer` |
| Continuous treatments | `cycle_dual`, `gnn_scm`, `jsbf`, `bridge_diff` |
| Density estimation | `flow_ssc`, `cnflow`, `m2_vae` |
| Transformer-based | `masked_tabular_transformer`, `tab_jepa`, `dag_transformer` |

---

## Understanding the XTY Framework

**XTY** represents the three core components in treatment effect estimation:
- **X**: Pre-treatment covariates (features describing each unit)
- **T**: Treatment assignment (discrete classes or continuous values)
- **Y**: Post-treatment outcomes (what we want to predict/understand)

All models in this library learn relationships between these components, with varying assumptions and capabilities.

### Treatment Label Conventions

- **Discrete treatments**: Integer values `0` to `k-1` where `k` is the number of treatment classes
- **Continuous treatments**: Set `k=None` when instantiating the model
- **Missing treatments**: Use `-1` to indicate unlabeled samples (handled automatically by trainers)

---

## Model Categories

### 1. Discriminative Models

These models directly predict outcomes from covariates and treatments. Best when you have clear prediction targets and want interpretable treatment effects.

#### `dragon_net` - DragonNet
**Best for**: Doubly robust treatment effect estimation with targeted regularization.

DragonNet uses a shared representation with separate outcome and propensity heads, plus a reconstruction head for p(t|x,y). Provides doubly robust effect estimation with uncertainty weighting.

```python
from xtylearner.models import create_model
model = create_model("dragon_net", d_x=10, d_y=1, k=2)
```

**Key parameters**:
- `hidden_dim`: Size of shared representation (default: 128)
- `alpha`: Weight for propensity loss

**When to use**: You need robust ATE/CATE estimates with theoretical guarantees.

---

#### `cacore` - Contrastive Causal Representation
**Best for**: Learning representations that maximize information about treatment-outcome relationships.

Uses InfoNCE contrastive objective to link the covariate representation h(x) with joint (y,t) embeddings.

```python
model = create_model("cacore", d_x=10, d_y=1, k=2)
```

**When to use**: You want to learn representations that capture causal structure.

---

#### `cycle_dual` - Cycle-Consistent Dual Network
**Best for**: Semi-supervised learning with continuous treatments.

Uses cycle consistency to recover covariates from predictions. Works well when treatment labels are missing.

```python
# For continuous treatment
model = create_model("cycle_dual", d_x=10, d_y=1, k=None)
# For discrete treatment
model = create_model("cycle_dual", d_x=10, d_y=1, k=3)
```

**When to use**: You have continuous treatments or significant missing treatment labels.

---

#### `multitask` - Multi-Task Self-Training
**Best for**: Simple semi-supervised baseline with pseudo-labeling.

Shared encoder with separate outcome/treatment heads. Generates pseudo-labels for unlabeled data iteratively.

```python
model = create_model("multitask", d_x=10, d_y=1, k=2)
```

**When to use**: You want a simple, interpretable semi-supervised approach.

---

#### `semiite` - Semi-Supervised ITE
**Best for**: Co-training approach for individual treatment effects.

Uses three outcome networks with reconstruction, MMD regularization, and KL divergence on propensity scores.

```python
model = create_model("semiite", d_x=10, d_y=1, k=2)
```

**When to use**: You need individual-level treatment effects with regularization.

---

### 2. Generative Models

These models learn the full joint distribution p(X,T,Y). Best for density estimation, sampling, and handling missing data.

#### `m2_vae` - Semi-Supervised M2 VAE
**Best for**: Semi-supervised learning with latent variables.

Classic M2 VAE architecture supporting both discrete and continuous treatments. Uses semi-supervised ELBO with labeled/unlabeled split.

```python
model = create_model("m2_vae", d_x=10, d_y=1, k=2, z_dim=16)
```

**Key parameters**:
- `z_dim`: Latent dimension (default: 16)
- `hidden_dim`: Hidden layer size

**When to use**: You want a well-understood generative baseline.

---

#### `ss_cevae` - Semi-Supervised CEVAE
**Best for**: Causal effect estimation with latent confounders.

Semi-supervised extension of the Conditional Effect VAE. Handles incomplete data with q(z|x,t,y) and q(t|x,y) encoders.

```python
model = create_model("ss_cevae", d_x=10, d_y=1, k=2)
```

**When to use**: You suspect latent confounders and have missing treatment labels.

---

#### `cevae_m` - CEVAE with Latent Treatment
**Best for**: Partially observed treatment labels with continuous latent treatment.

CEVAE variant that models treatment as a latent variable.

```python
model = create_model("cevae_m", d_x=10, d_y=1, k=2)
```

**When to use**: Treatment is partially observed and may have measurement error.

---

#### `joint_ebm` - Joint Energy-Based Model
**Best for**: Flexible joint density modeling.

Models full joint p(x,t,y) using an energy function. Uses contrastive loss on labeled rows and marginalized loss on unlabeled.

```python
model = create_model("joint_ebm", d_x=10, d_y=1, k=2)
```

**When to use**: You want flexible density estimation without distributional assumptions.

---

#### `scgm` - Semi-Supervised Causal Generative Model
**Best for**: Generative modeling with explicit causal structure.

Learns a generative model that respects causal structure, separating confounding effects.

```python
model = create_model("scgm", d_x=10, d_y=1, k=2, z_dim=8)
```

**When to use**: You want to explicitly model confounding in a generative framework.

---

### 3. Flow-Based Models

Normalizing flows provide exact density estimation through invertible transformations.

#### `flow_ssc` - Semi-Supervised Conditional Flows
**Best for**: Semi-supervised learning with exact likelihoods.

Uses RealNVP architecture with semi-supervised ELBO. Handles missing treatments naturally.

```python
model = create_model("flow_ssc", d_x=10, d_y=1, k=2)
```

**When to use**: You need exact density evaluation and semi-supervised learning.

---

#### `cnflow` - Conditional Normalizing Flow
**Best for**: Modeling conditional outcome distributions p(y|x,t).

Uses masked autoregressive transforms for flexible conditional density estimation.

```python
model = create_model("cnflow", d_x=10, d_y=1, k=2)
```

**When to use**: You need full outcome distributions, not just point predictions.

---

#### `lt_flow_diff` - Latent Flow with Diffusion Prior
**Best for**: Combining flow flexibility with diffusion priors.

Combines conditional flow with latent diffusion prior for both continuous and discrete treatments.

```python
model = create_model("lt_flow_diff", d_x=10, d_y=1, k=2)
```

**When to use**: You want state-of-the-art density estimation with diffusion.

---

### 4. Diffusion Models

Score-based and denoising diffusion models for generation and imputation.

#### `jsbf` - Joint Score-Based Factorization
**Best for**: Flexible diffusion-based joint modeling.

Score-based diffusion of the full joint (x,t,y). Handles missing treatments naturally through score matching.

```python
model = create_model("jsbf", d_x=10, d_y=1, k=2)
```

**Key parameters**:
- `n_steps`: Number of diffusion steps
- `hidden_dim`: Network hidden dimension

**When to use**: You want flexible diffusion-based density estimation with missing data support.

---

#### `bridge_diff` - Diffusion Bridge for Counterfactuals
**Best for**: Counterfactual outcome estimation.

Couples Y(0) and Y(1) counterfactual draws using a score-based bridge architecture.

```python
model = create_model("bridge_diff", d_x=10, d_y=1, k=2)
```

**When to use**: You specifically need counterfactual outcome distributions.

---

#### `diffusion_cevae` - Diffusion-Based CEVAE
**Best for**: Combining CEVAE with diffusion dynamics.

Diffusion variant of CEVAE using latent score matching with semi-supervised objective.

```python
model = create_model("diffusion_cevae", d_x=10, d_y=1, k=2)
```

**When to use**: You want CEVAE's causal framework with diffusion's flexibility.

---

#### `eg_ddi` - Energy-Guided Discrete Diffusion Imputer
**Best for**: Discrete treatment imputation.

Uses discrete diffusion for treatment imputation with energy guidance on outcome predictions.

```python
model = create_model("eg_ddi", d_x=10, d_y=1, k=2)
```

**When to use**: You have discrete treatments with many missing labels.

---

#### `ctm_t` - Consistency-Trajectory Diffusion
**Best for**: Fast diffusion sampling with consistency training.

Consistency trajectory diffusion with treatment head and mixture guidance objectives.

```python
model = create_model("ctm_t", d_x=10, d_y=1, k=2)
```

**When to use**: You need fast diffusion sampling at inference time.

---

### 5. Structural Causal Models

Models that explicitly learn causal structure and relationships.

#### `gnn_scm` - Graph Neural Structural Causal Model
**Best for**: Learning causal DAG structure from data.

Learns a directed acyclic graph (DAG) adjacency matrix with neural structural equations. Uses NOTEARS acyclicity constraint.

```python
model = create_model("gnn_scm", d_x=10, d_y=1, k=2)
```

**Key parameters**:
- `lambda_acyc`: Acyclicity penalty strength
- `gamma_l1`: L1 regularization on adjacency matrix
- `forbid_y_to_x`: Exclude edges from Y to X (default: True)

**When to use**: You want to discover causal structure while estimating effects.

---

#### `diffusion_gnn_scm` - Diffusion-Based GNN-SCM
**Best for**: Causal structure learning with diffusion.

Diffusion variant of GNN-SCM with score matching objective.

```python
model = create_model("diffusion_gnn_scm", d_x=10, d_y=1, k=2)
```

**When to use**: You want GNN-SCM's structure learning with diffusion's flexibility.

---

#### `gnn_ebm` - Energy-Based Graph Neural Model
**Best for**: Simpler inference than structural equations.

Energy-based variant of GNN-SCM with simpler inference procedure.

```python
model = create_model("gnn_ebm", d_x=10, d_y=1, k=2)
```

**When to use**: You want structure learning with energy-based modeling.

---

#### `deconfounder_cfm` - Causal Factor Model
**Best for**: Multiple correlated treatments with confounding.

Two-stage approach: (1) learn substitute confounder, (2) predict outcomes. Uses HSIC-based confounding measurement.

```python
model = create_model("deconfounder_cfm", d_x=10, d_y=1, k=2)
```

**When to use**: You have multiple treatments that may share common confounders.

---

### 6. Transformer-Based Models

Modern transformer architectures adapted for tabular causal inference.

#### `masked_tabular_transformer` - Masked Token Transformer
**Best for**: Self-supervised pre-training on tabular data.

Transformer encoder with masked-token training objective. Handles mixed modalities (X, T, Y).

```python
model = create_model("masked_tabular_transformer", d_x=10, d_y=1, k=2)
```

**When to use**: You have large tabular datasets and want pre-training benefits.

---

#### `tab_jepa` - Joint-Embedding Predictive Architecture
**Best for**: Learning transferable representations.

Masks columns of (X, T, Y) and predicts latent representations using momentum-based target encoder.

```python
model = create_model("tab_jepa", d_x=10, d_y=1, k=2)
```

**When to use**: You want representations that transfer across tasks.

---

#### `dag_transformer` - DAG-Aware Transformer
**Best for**: Incorporating known causal structure into transformers.

Attention mechanism that respects causal DAG structure. Provides flexible ATE/CATE estimation.

```python
model = create_model("dag_transformer", d_x=10, d_y=1, k=2)
```

**When to use**: You have prior knowledge about causal structure.

---

### 7. Semi-Supervised Baselines

Well-established semi-supervised methods adapted for treatment effect estimation.

#### `lp_knn` - Label Propagation (k-NN)
**Best for**: Fast, non-parametric baseline.

k-NN based label propagation with optional scikit-learn regressor. No neural network training required.

```python
model = create_model("lp_knn", d_x=10, d_y=1, k=2)
```

**When to use**: Quick baseline or when you have limited compute.

---

#### `mean_teacher` - Exponential Moving Average Teacher
**Best for**: Stable semi-supervised training.

EMA teacher-student consistency with ramp-up scheduling.

```python
model = create_model("mean_teacher", d_x=10, d_y=1, k=2)
```

**Key parameters**:
- `ema_decay`: EMA coefficient (0.99-0.999)
- `ramp_up`: Epochs to ramp up consistency weight

**When to use**: You want stable, well-understood semi-supervised learning.

---

#### `vat` - Virtual Adversarial Training
**Best for**: Consistency regularization without data augmentation.

Adversarial perturbations for consistency regularization. Domain-agnostic (no special augmentations needed).

```python
model = create_model("vat", d_x=10, d_y=1, k=2)
```

**When to use**: You want semi-supervised learning without designing augmentations.

---

#### `fixmatch` - Pseudo-Labeling with Strong Augmentation
**Best for**: High-confidence pseudo-labeling.

Combines pseudo-labels with confidence thresholding and weak/strong augmentation pairs.

```python
model = create_model("fixmatch", d_x=10, d_y=1, k=2)
```

**When to use**: You can define meaningful augmentations for your data.

---

#### `vime` - Variational Information Masking
**Best for**: Two-stage self-supervised approach.

Pre-training via mask-and-corrupt, then fine-tuning. Learns feature importance weighting.

```python
model = create_model("vime", d_x=10, d_y=1, k=2)
```

**When to use**: You have abundant unlabeled data for pre-training.

---

#### `ganite` - GAN-based Individual Treatment Effects
**Best for**: Adversarial treatment effect estimation.

Binary-treatment GAN with generator and discriminator for robust ITE estimation.

```python
model = create_model("ganite", d_x=10, d_y=1, k=2)
```

**When to use**: Binary treatment with adversarial robustness needs.

---

#### `ss_dml` - Semi-Supervised Double Machine Learning
**Best for**: Debiased treatment effect estimation.

Uses double/debiased machine learning framework. Requires `xtylearner[causal]` extra.

```bash
pip install xtylearner[causal]
```

```python
model = create_model("ss_dml", d_x=10, d_y=1, k=2)
```

**When to use**: You need theoretically grounded debiased estimates.

---

### 8. Other Specialized Models

#### `crf` / `crf_discrete` - Conditional Random Field
**Best for**: Structured prediction with dependencies.

CRF models for continuous and discrete outcomes respectively.

```python
model = create_model("crf", d_x=10, d_y=1, k=2)  # Continuous
model = create_model("crf_discrete", d_x=10, d_y=1, k=2)  # Discrete
```

---

#### `em` - Expectation-Maximization Model
**Best for**: Classical EM-based semi-supervised learning.

EM algorithm for handling missing treatment labels.

```python
model = create_model("em", d_x=10, d_y=1, k=2)
```

---

#### `ccl_cpc` - Contrastive Predictive Coding
**Best for**: Sequential covariates with temporal structure.

Handles sequential data with partially observed labels.

```python
model = create_model("ccl_cpc", d_x=10, d_y=1, k=2)
```

---

#### `vacim` - CEVAE with Conditional Masking
**Best for**: Robust handling of missing data.

CEVAE variant with conditional input masking.

```python
model = create_model("vacim", d_x=10, d_y=1, k=2)
```

---

#### `cycle_vat` - Cycle Consistency + VAT
**Best for**: Combining cycle consistency with adversarial training.

Forward/inverse classifiers with VAT regularization and posterior mixing.

```python
model = create_model("cycle_vat", d_x=10, d_y=1, k=2)
```

---

#### `gflownet_treatment` - Generative Flow Network
**Best for**: Sampling treatments proportional to outcome likelihood.

Use for counterfactual reasoning and treatment optimization.

```python
model = create_model("gflownet_treatment", d_x=10, d_y=1, k=2)
```

---

#### `factor_vae_plus` - Multi-Treatment Factor VAE
**Best for**: Multiple categorical treatments with disentanglement.

Handles multiple treatment dimensions with disentangled latent factors.

```python
model = create_model("factor_vae_plus", d_x=10, d_y=1, k=[2, 3])  # Multiple treatments
```

---

#### `prob_circuit` - Probabilistic Sum-Product Network
**Best for**: Exact marginalization in density estimation.

Uses SPFlow for structured density estimation with tractable inference.

```python
model = create_model("prob_circuit", d_x=10, d_y=1, k=2)
```

---

## Decision Flowchart

Use this flowchart to guide your model selection:

```
START
  │
  ├─► Do you have missing treatment labels?
  │     │
  │     ├─► Yes ──► Do you want to learn causal structure?
  │     │           │
  │     │           ├─► Yes ──► gnn_scm, diffusion_gnn_scm
  │     │           │
  │     │           └─► No ──► Do you need density estimation?
  │     │                       │
  │     │                       ├─► Yes ──► jsbf, bridge_diff, ss_cevae
  │     │                       │
  │     │                       └─► No ──► multitask, mean_teacher, vat
  │     │
  │     └─► No ──► Do you need causal effect estimates?
  │                 │
  │                 ├─► Yes ──► dragon_net, cacore, deconfounder_cfm
  │                 │
  │                 └─► No ──► Any supervised model
  │
  ├─► Are treatments continuous?
  │     │
  │     └─► Yes ──► cycle_dual, gnn_scm, jsbf (with k=None)
  │
  ├─► Do you have sequential/temporal data?
  │     │
  │     └─► Yes ──► ccl_cpc
  │
  └─► Do you need fast inference?
        │
        ├─► Yes ──► lp_knn, multitask
        │
        └─► No ──► Consider diffusion models for best quality
```

---

## Common Usage Patterns

### Basic Training

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from xtylearner.models import create_model
from xtylearner.training import train_model

# Prepare data
X = torch.randn(1000, 10)  # Covariates
T = torch.randint(0, 2, (1000,))  # Binary treatment
Y = torch.randn(1000, 1)  # Outcomes

# Mark some treatments as missing
T[800:] = -1  # Last 200 samples unlabeled

dataset = TensorDataset(X, Y, T)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Create and train model
model = create_model("dragon_net", d_x=10, d_y=1, k=2)
train_model(model, loader, epochs=100)
```

### Continuous Treatment

```python
# For continuous treatments, set k=None
model = create_model("cycle_dual", d_x=10, d_y=1, k=None)

# Treatment values can be any float
T = torch.randn(1000)  # Continuous treatment
```

### Treatment Effect Estimation

```python
# After training, estimate effects
model.eval()
with torch.no_grad():
    # Predict outcomes under different treatments
    y_t0 = model(X, torch.zeros(len(X)))
    y_t1 = model(X, torch.ones(len(X)))

    # Individual treatment effects
    ite = y_t1 - y_t0

    # Average treatment effect
    ate = ite.mean()
```

---

## Performance Considerations

| Model Type | Training Speed | Inference Speed | Memory Usage |
|------------|----------------|-----------------|--------------|
| `lp_knn` | Fast | Fast | Low |
| Discriminative | Fast | Fast | Medium |
| VAE-based | Medium | Fast | Medium |
| Flow-based | Medium | Medium | Medium |
| Diffusion | Slow | Slow | High |
| GNN-SCM | Slow | Medium | Medium |
| Transformers | Medium | Fast | High |

---

## Further Reading

- Individual model documentation in `docs/baselines/` and `docs/novel/`
- [AGENTS.md](../AGENTS.md) for contributing new models
- API reference for detailed parameter descriptions
