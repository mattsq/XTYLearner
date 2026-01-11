# Ordinal Classification Support in XTYLearner

This document outlines a plan for adding ordinal classification support to XTYLearner, enabling the package to handle treatment variables with natural ordering (e.g., severity levels: none < mild < moderate < severe).

---

## 1. Motivation

### What is Ordinal Classification?

Ordinal classification (also called ordinal regression) handles categorical variables where the classes have a **natural ordering** but the distances between classes are not necessarily equal or known. Examples include:

- **Disease severity**: none < mild < moderate < severe
- **Dosage levels**: low < medium < high
- **Treatment intensity**: 0 < 1 < 2 < 3 < 4
- **Quality ratings**: poor < fair < good < excellent
- **Education levels**: high school < bachelor's < master's < PhD

### Why It Matters for Causal Inference

In treatment effect estimation, ordinal treatments are common:
- Drug dosage levels (treatment intensity)
- Intervention intensities in behavioral studies
- Staged medical procedures
- Policy intensity variations

Standard categorical classification ignores ordering information, leading to:
1. **Suboptimal predictions**: Predicting class 0 when truth is class 3 is treated the same as predicting class 2
2. **Information loss**: The model doesn't learn that adjacent classes are more similar
3. **Poor calibration**: Probability estimates don't respect ordinal structure

---

## 2. Current State of XTYLearner

### Treatment Classification Approach

Currently, all models treat discrete treatments as **unordered categorical variables**:

```python
# Current pattern in all models
logits_T = self.head_T(features)  # shape: (batch, k)
loss = F.cross_entropy(logits_T, T_target)  # Multinomial loss
probs = F.softmax(logits_T, dim=-1)  # Standard softmax
prediction = logits_T.argmax(dim=-1)  # Point prediction
```

### Key Locations for Modification

| Component | Location | Current Behavior |
|-----------|----------|------------------|
| Loss functions | `xtylearner/losses.py` | Cross-entropy only |
| Treatment heads | `xtylearner/models/*.py` | Linear → k logits |
| Metrics | `xtylearner/training/metrics.py` | Accuracy via argmax |
| Base trainer | `xtylearner/training/base_trainer.py` | Accuracy-based evaluation |

---

## 3. Proposed Ordinal Classification Methods

We recommend implementing multiple ordinal classification approaches to support different use cases:

### 3.1 Cumulative Link Models (Proportional Odds)

The classic statistical approach. Model cumulative probabilities:

$$P(T \leq j | X) = \sigma(\theta_j - f(X))$$

where $\theta_1 < \theta_2 < ... < \theta_{k-1}$ are ordered thresholds and $f(X)$ is a shared representation.

**Pros**: Well-understood, interpretable, respects ordinality
**Cons**: Proportional odds assumption may not hold

**Implementation**:
```python
class CumulativeLinkHead(nn.Module):
    """Ordinal head using cumulative link model."""
    def __init__(self, in_features: int, k: int, link: str = "logit"):
        # k-1 ordered thresholds
        self.thresholds = nn.Parameter(torch.linspace(-2, 2, k - 1))
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        # Returns cumulative probabilities P(T <= j)
        logits = self.fc(x)  # (batch, 1)
        cumprobs = torch.sigmoid(self.thresholds - logits)
        return cumprobs  # (batch, k-1)
```

### 3.2 CORAL (Consistent Rank Logits)

From Cao et al. (2020). Uses k-1 binary classifiers sharing weights:

$$P(T > j | X) = \sigma(f(X) - b_j)$$

**Pros**: Simple extension of binary classification, no proportional odds assumption
**Cons**: Requires careful weight sharing

**Implementation**:
```python
class CORALHead(nn.Module):
    """CORAL: Consistent Rank Logits for ordinal regression."""
    def __init__(self, in_features: int, k: int):
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(k - 1))

    def forward(self, x):
        logits = self.fc(x)  # (batch, 1)
        return logits - self.biases  # (batch, k-1)
```

### 3.3 Soft Label Encoding

Encode ordinal structure through label smoothing that respects adjacency:

$$\text{soft\_label}[j] = \exp(-\lambda |j - t|)$$

**Pros**: Works with existing cross-entropy, easy to implement
**Cons**: Hyperparameter sensitive

### 3.4 Ordinal Regression Loss (Unimodal)

Train with soft targets encouraging unimodal distributions centered on the true class:

$$L = -\sum_j w_j \log p_j, \quad w_j = \exp(-\alpha|j - t|^2)$$

**Pros**: Probabilistic interpretation, smooth gradients
**Cons**: Additional hyperparameters

### 3.5 Ranking Losses

Pairwise ranking loss ensuring correct ordering:

$$L = \sum_{i: T_i > T_j} \max(0, m - (s_i - s_j))$$

**Pros**: Directly optimizes ranking, no distributional assumptions
**Cons**: O(n²) complexity, may need sampling

---

## 4. Implementation Plan

### Phase 1: Core Infrastructure ✅ COMPLETED

#### 4.1 Add Ordinal Loss Functions (`losses.py`)

```python
# New functions to add:
def cumulative_link_loss(cumprobs, target, k):
    """NLL for cumulative link model."""

def coral_loss(logits, target, k):
    """CORAL binary cross-entropy loss."""

def ordinal_regression_loss(logits, target, alpha=1.0):
    """Soft-label ordinal loss with Gaussian weighting."""

def ordinal_focal_loss(logits, target, gamma=2.0):
    """Focal loss variant for ordinal classification."""
```

#### 4.2 Create Ordinal Heads (`models/heads.py`)

New module for ordinal-specific prediction heads:

```python
class OrdinalHead(nn.Module):
    """Factory for ordinal prediction heads."""

    def __init__(self, in_features, k, method="cumulative"):
        if method == "cumulative":
            self.head = CumulativeLinkHead(in_features, k)
        elif method == "coral":
            self.head = CORALHead(in_features, k)
        elif method == "standard":
            self.head = nn.Linear(in_features, k)
```

#### 4.3 Add Ordinal Metrics (`training/metrics.py`)

```python
def ordinal_mae(predictions, targets):
    """Mean Absolute Error treating classes as integers."""

def ordinal_rmse(predictions, targets):
    """Root Mean Squared Error on class indices."""

def quadratic_weighted_kappa(predictions, targets, k):
    """Cohen's Kappa with quadratic weights."""

def ordinal_accuracy(predictions, targets, tolerance=0):
    """Accuracy allowing off-by-tolerance errors."""

def adjacent_accuracy(predictions, targets):
    """Accuracy counting adjacent predictions as correct."""

def spearman_correlation(predictions, targets):
    """Spearman rank correlation coefficient."""
```

### Phase 2: Model Integration

#### 4.4 Add Ordinal Flag to Model Registry

Update model instantiation to support ordinal mode:

```python
# In get_model() / create_model()
model = create_model(
    "dragon_net",
    d_x=10,
    d_y=1,
    k=5,
    ordinal=True,  # New parameter
    ordinal_method="coral"  # "cumulative", "coral", "soft_label"
)
```

#### 4.5 Update Key Models

Priority models to update (high impact, representative architectures):

1. **`dragon_net`** - Discriminative baseline with propensity/outcome heads
2. **`cycle_dual`** - Semi-supervised with cycle consistency
3. **`multitask`** - Simple multi-task baseline
4. **`ss_cevae`** - Generative model with treatment encoder
5. **`jsbf`** - Diffusion model with discrete treatment handling

For each model:
- Add `ordinal: bool = False` parameter to `__init__`
- Replace treatment head with `OrdinalHead` when `ordinal=True`
- Update `loss()` to use appropriate ordinal loss
- Update `predict_treatment_proba()` to return proper ordinal probabilities

#### 4.6 Create Ordinal-Specific Model Variants (Optional)

For maximum flexibility, create dedicated ordinal variants:

```python
@register_model("ordinal_dragon_net")
class OrdinalDragonNet(DragonNet):
    """DragonNet with built-in ordinal classification."""
```

### Phase 3: Training Integration

#### 4.7 Update Base Trainer

Modify `base_trainer.py` to compute ordinal metrics:

```python
def _treatment_metrics(self, logits, targets, ordinal=False):
    metrics = {}
    if ordinal:
        preds = self._ordinal_predict(logits)
        metrics["treatment_mae"] = ordinal_mae(preds, targets)
        metrics["treatment_qwk"] = quadratic_weighted_kappa(preds, targets)
        metrics["treatment_adjacent_acc"] = adjacent_accuracy(preds, targets)
    else:
        # Existing accuracy computation
        metrics["treatment_accuracy"] = accuracy(logits, targets)
    return metrics
```

#### 4.8 Add Ordinal Trainer (Optional)

For complex ordinal-specific training schemes:

```python
class OrdinalTrainer(SupervisedTrainer):
    """Specialized trainer for ordinal classification models."""

    def __init__(self, model, optimizer, loader,
                 ordinal_weight=1.0, unimodal_reg=0.1):
        super().__init__(model, optimizer, loader)
        self.ordinal_weight = ordinal_weight
        self.unimodal_reg = unimodal_reg
```

### Phase 4: Utilities and Testing

#### 4.9 Probability Conversion Utilities

```python
def cumulative_to_class_probs(cumprobs):
    """Convert P(T<=j) to P(T=j)."""
    # P(T=0) = P(T<=0)
    # P(T=j) = P(T<=j) - P(T<=j-1) for j > 0

def class_probs_to_cumulative(probs):
    """Convert P(T=j) to P(T<=j)."""

def ordinal_predict(cumprobs, method="median"):
    """Predict ordinal class from cumulative probabilities.

    Methods:
    - "median": Class where cumprob crosses 0.5
    - "mean": Expected value E[T]
    - "mode": Most likely class
    """
```

#### 4.10 Testing

Add comprehensive tests:

```python
# tests/test_ordinal.py
def test_cumulative_link_loss():
    """Test cumulative link loss computation."""

def test_coral_loss():
    """Test CORAL loss computation."""

def test_ordinal_metrics():
    """Test MAE, QWK, adjacent accuracy."""

def test_ordinal_model_training():
    """End-to-end ordinal model training."""

def test_cumulative_probability_conversion():
    """Test probability space conversions."""
```

---

## 5. API Design

### User-Facing API

```python
from xtylearner.models import create_model
from xtylearner.training import Trainer

# Create ordinal model
model = create_model(
    "dragon_net",
    d_x=10,
    d_y=1,
    k=5,  # 5 ordinal levels
    ordinal=True,
    ordinal_method="coral"  # or "cumulative", "soft_label"
)

# Training is identical
trainer = Trainer(model, optimizer, loader)
trainer.fit(epochs=100)

# Evaluation returns ordinal metrics
metrics = trainer.evaluate(test_loader)
# metrics now includes:
# - treatment_mae: Mean Absolute Error on ranks
# - treatment_qwk: Quadratic Weighted Kappa
# - treatment_adjacent_acc: Adjacent-class accuracy
# - treatment_accuracy: Exact accuracy (existing)
```

### Low-Level API

```python
from xtylearner.losses import coral_loss, cumulative_link_loss
from xtylearner.training.metrics import ordinal_mae, quadratic_weighted_kappa

# Direct loss computation
loss = coral_loss(logits, targets, k=5)

# Direct metric computation
mae = ordinal_mae(predictions, targets)
qwk = quadratic_weighted_kappa(predictions, targets, k=5)
```

---

## 6. File Changes Summary

| File | Changes |
|------|---------|
| `xtylearner/losses.py` | Add ordinal loss functions |
| `xtylearner/models/heads.py` | New file: ordinal prediction heads |
| `xtylearner/models/components.py` | Import/export ordinal heads |
| `xtylearner/models/dragon_net.py` | Add ordinal support |
| `xtylearner/models/cycle_dual.py` | Add ordinal support |
| `xtylearner/models/multitask.py` | Add ordinal support |
| `xtylearner/training/metrics.py` | Add ordinal metrics |
| `xtylearner/training/base_trainer.py` | Compute ordinal metrics |
| `tests/test_ordinal.py` | New file: ordinal tests |
| `docs/model_guide.md` | Document ordinal usage |

---

## 7. Considerations and Trade-offs

### Method Selection Guide

| Method | Best For | Assumptions | Complexity |
|--------|----------|-------------|------------|
| Cumulative Link | Traditional statistics, interpretability | Proportional odds | Low |
| CORAL | Deep learning, no assumptions | None | Low |
| Soft Labels | Quick integration with existing models | Gaussian-like errors | Very Low |
| Ranking Loss | When relative ordering matters most | None | Medium |

### Backward Compatibility

- All changes are **additive** - existing code continues to work
- `ordinal=False` (default) maintains current behavior
- Models without ordinal support raise clear errors if `ordinal=True`

### Performance Considerations

- Ordinal heads add minimal overhead (k-1 vs k outputs)
- Metrics computation is O(n) - negligible cost
- No architectural changes to backbone networks

---

## 8. Future Extensions

### Potential Additions

1. **Deep ordinal regression** - Multi-layer ordinal heads with shared representations
2. **Ordinal calibration** - Temperature scaling for ordinal probabilities
3. **Ordinal uncertainty** - Confidence intervals respecting ordinal structure
4. **Multi-output ordinal** - Multiple ordinal variables (e.g., severity + duration)
5. **Ordinal diffusion** - Native ordinal support in diffusion models
6. **Ordinal active learning** - Query strategies considering ordinal structure

### Research Directions

- **Ordinal causal effects**: ATE/CATE for ordinal treatments with proper effect size interpretation
- **Ordinal counterfactuals**: What-if analysis respecting ordinal constraints
- **Ordinal semi-supervised**: Leverage ordinality in unlabeled data

---

## 9. References

1. **McCullagh, P. (1980)**. Regression models for ordinal data. *Journal of the Royal Statistical Society B*.
2. **Cao, W., Mirjalili, V., & Raschka, S. (2020)**. Rank consistent ordinal regression for neural networks with application to age estimation. *Pattern Recognition Letters*.
3. **Cheng, J., Wang, Z., & Pollastri, G. (2008)**. A neural network approach to ordinal regression. *IJCNN*.
4. **Herbrich, R., Graepel, T., & Obermayer, K. (1999)**. Large margin rank boundaries for ordinal regression. *Advances in Large Margin Classifiers*.
5. **Frank, E., & Hall, M. (2001)**. A simple approach to ordinal classification. *ECML*.

---

## 10. Summary

Adding ordinal classification support to XTYLearner involves:

1. **New loss functions** in `losses.py` for ordinal objectives
2. **New prediction heads** for cumulative/CORAL approaches
3. **New metrics** for ordinal-specific evaluation
4. **Model updates** to support `ordinal=True` flag
5. **Trainer updates** to compute ordinal metrics

The implementation follows XTYLearner's existing patterns (registry, modular losses, trainer factories) and maintains full backward compatibility. Users can enable ordinal mode with a single flag while getting appropriate losses and metrics automatically.
