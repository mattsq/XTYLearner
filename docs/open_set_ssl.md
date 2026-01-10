# Open-Set Semi-Supervised Learning: Handling OOD Unlabelled Data

This document provides theoretical background and implementation guidance for handling **out-of-distribution (OOD) samples in unlabelled data** within the XTYLearner framework. When unlabelled treatment data contains samples from treatments not present in the labelled set, standard SSL methods can fail catastrophically—this guide presents principled solutions from the literature.

---

## 1. Problem Statement

### 1.1 The Open-Set SSL Challenge

In standard semi-supervised learning, we assume labelled and unlabelled data share the same label space (treatment set). However, in practice:

- Unlabelled data may contain **novel treatments** not seen in the labelled set
- Some samples may be **out-of-distribution** due to covariate shift
- Treatment assignment mechanisms may differ between labelled and unlabelled populations

This scenario is called **Open-Set Semi-Supervised Learning (OSSL)**.

### 1.2 Why Standard SSL Fails

Standard methods like Mean Teacher and FixMatch suffer from several failure modes:

| Failure Mode | Mechanism | Citation |
|--------------|-----------|----------|
| **Confirmation Bias** | Pseudo-labels on OOD samples reinforce incorrect predictions, creating a vicious cycle | Arazo et al., 2020 [1] |
| **Entropy Minimization Collapse** | SSL pushes decision boundaries to low-density regions; OOD samples force overconfident predictions causing collapse | Context-guided Entropy Minimization [2] |
| **Pseudo-Label Imbalance** | FixMatch generates imbalanced pseudo-labels when OOD data is present, destabilising training | Chen et al., 2023 [3] |

**Key finding**: Exploiting inconsistent unlabelled data can cause performance degradation *worse than supervised-only baselines* (Guo et al., 2024 [4]).

---

## 2. Literature Overview

### 2.1 Taxonomy of Approaches

```
Open-Set SSL Methods
├── Detection & Filtering
│   ├── OpenMatch (OVA + SOCR)           [Section 3]
│   ├── UASD (Uncertainty-Aware)         [Section 4]
│   └── DS³L (Distribution-Aware)        [Section 5]
├── Soft Weighting
│   ├── Confidence-Based Weighting       [Section 6]
│   └── Energy-Based Weighting           [Section 7]
├── Architecture Solutions
│   └── Clone (Closed Loop Networks)     [Section 8]
└── Dataset Selection
    └── MixMOOD (Ante-hoc Selection)     [Section 9]
```

---

## 3. OpenMatch: One-vs-All Outlier Detection

**Citation**: Saito, K., Kim, D., & Saenko, K. (2021). OpenMatch: Open-set Consistency Regularization for Semi-supervised Learning with Outliers. *NeurIPS 2021*. [5]

### 3.1 Core Idea

Train a **One-vs-All (OVA) classifier** alongside the standard classifier. Each OVA head learns to distinguish "this is class k" vs "this is not class k". A sample is marked as OOD only if **all** OVA classifiers reject it.

### 3.2 Mathematical Formulation

For K known classes, the OVA classifier has K binary outputs:

$$h_k(x) = \sigma(g_k(x)) \in [0, 1]$$

where $g_k$ is the k-th OVA logit. The OOD score is:

$$s_{\text{OOD}}(x) = 1 - \max_k h_k(x)$$

A sample is considered OOD if $s_{\text{OOD}}(x) > \tau_{\text{ova}}$.

### 3.3 Soft Open-Set Consistency Regularization (SOCR)

Rather than hard filtering, OpenMatch applies consistency regularization to the OVA outputs:

$$\mathcal{L}_{\text{SOCR}} = \frac{1}{|B_U|} \sum_{x \in B_U} \| h(\tilde{x}_1) - h(\tilde{x}_2) \|_2^2$$

where $\tilde{x}_1, \tilde{x}_2$ are two augmented views of $x$.

### 3.4 Implementation for XTYLearner

```python
# xtylearner/models/openmatch.py
from __future__ import annotations

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model
from .utils import ramp_up_sigmoid


@register_model("openmatch")
class OpenMatch(nn.Module):
    """OpenMatch: Open-set SSL with OVA outlier detection.

    Combines FixMatch-style pseudo-labeling with One-vs-All OOD detection
    and Soft Open-Set Consistency Regularization (SOCR).

    References
    ----------
    Saito et al., "OpenMatch: Open-set Consistency Regularization for
    Semi-supervised Learning with Outliers", NeurIPS 2021.
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int = 2,
        *,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: callable | None = None,
        residual: bool = False,
        # FixMatch params
        tau: float = 0.95,
        lambda_u: float = 1.0,
        # OVA params
        tau_ova: float = 0.5,
        lambda_socr: float = 0.5,
        # Ramp-up
        ramp_up: int = 40,
    ) -> None:
        super().__init__()
        self.k = k
        self.tau = tau
        self.lambda_u = lambda_u
        self.tau_ova = tau_ova
        self.lambda_socr = lambda_socr
        self.ramp_up = ramp_up
        self.step_count = 0

        # Outcome network
        self.outcome = make_mlp(
            [d_x + k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )

        # Shared backbone for treatment classifier
        self.backbone = make_mlp(
            [d_x + d_y, *hidden_dims],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )

        # Standard K-way classifier head
        self.classifier = nn.Linear(hidden_dims[-1], k)

        # OVA head: K binary classifiers (sigmoid outputs)
        self.ova_head = nn.Linear(hidden_dims[-1], k)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict outcome y from covariates x and treatment t."""
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))

    def _get_ova_scores(self, inp: torch.Tensor) -> torch.Tensor:
        """Get OVA confidence scores (K sigmoids)."""
        features = self.backbone(inp)
        return torch.sigmoid(self.ova_head(features))

    def _get_class_logits(self, inp: torch.Tensor) -> torch.Tensor:
        """Get K-way classification logits."""
        features = self.backbone(inp)
        return self.classifier(features)

    def _is_ood(self, ova_scores: torch.Tensor) -> torch.Tensor:
        """Detect OOD samples: OOD if max OVA score < threshold."""
        max_ova, _ = ova_scores.max(dim=-1)
        return max_ova < self.tau_ova

    def _add_noise(self, x: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
        """Add Gaussian noise for consistency regularization."""
        if self.training:
            return x + scale * torch.randn_like(x)
        return x

    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute OpenMatch loss with OOD filtering and SOCR."""
        labelled = t_obs >= 0
        loss = torch.tensor(0.0, device=x.device)

        inp = torch.cat([x, y], dim=-1)

        # === Supervised losses on labelled data ===
        if labelled.any():
            t_use = t_obs[labelled].clamp_min(0)
            t_onehot = F.one_hot(t_use.to(torch.long), self.k).float()

            # Outcome loss
            y_hat = self.outcome(torch.cat([x[labelled], t_onehot], dim=-1))
            loss += F.mse_loss(y_hat, y[labelled])

            # Classification loss
            logits = self._get_class_logits(inp[labelled])
            loss += F.cross_entropy(logits, t_use)

            # OVA supervised loss (binary CE for each class)
            ova_scores = self._get_ova_scores(inp[labelled])
            ova_targets = F.one_hot(t_use.to(torch.long), self.k).float()
            loss += F.binary_cross_entropy(ova_scores, ova_targets)

        # === Unsupervised losses on unlabelled data ===
        if (~labelled).any():
            inp_u = inp[~labelled]

            # Get OVA scores and filter OOD samples
            with torch.no_grad():
                ova_scores = self._get_ova_scores(inp_u)
                is_ood = self._is_ood(ova_scores)
                is_inlier = ~is_ood

            # FixMatch on inliers only
            if is_inlier.any():
                inp_inlier = inp_u[is_inlier]

                # Weak augmentation -> pseudo-label
                with torch.no_grad():
                    logits_weak = self._get_class_logits(inp_inlier)
                    probs = logits_weak.softmax(dim=-1)
                    max_prob, pseudo_label = probs.max(dim=-1)
                    mask = max_prob >= self.tau

                # Strong augmentation -> cross-entropy
                if mask.any():
                    inp_strong = self._add_noise(inp_inlier[mask], scale=0.2)
                    logits_strong = self._get_class_logits(inp_strong)
                    L_unsup = F.cross_entropy(logits_strong, pseudo_label[mask])
                    lam = ramp_up_sigmoid(self.step_count, self.ramp_up, self.lambda_u)
                    loss += lam * L_unsup

            # SOCR: Soft Open-Set Consistency Regularization (on ALL unlabelled)
            inp_u1 = self._add_noise(inp_u, scale=0.1)
            inp_u2 = self._add_noise(inp_u, scale=0.1)
            ova_1 = self._get_ova_scores(inp_u1)
            ova_2 = self._get_ova_scores(inp_u2)
            L_socr = F.mse_loss(ova_1, ova_2)
            loss += self.lambda_socr * L_socr

        return loss

    def step(self) -> None:
        """Increment step counter after optimizer step."""
        self.step_count += 1

    @torch.no_grad()
    def predict_treatment_proba(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Predict treatment probabilities (for inliers only)."""
        inp = torch.cat([x, y], dim=-1)
        logits = self._get_class_logits(inp)
        return logits.softmax(dim=-1)

    @torch.no_grad()
    def predict_ood_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return OOD score: higher = more likely OOD."""
        inp = torch.cat([x, y], dim=-1)
        ova_scores = self._get_ova_scores(inp)
        return 1.0 - ova_scores.max(dim=-1).values

    @torch.no_grad()
    def predict_outcome(
        self, x: torch.Tensor, t: int | torch.Tensor
    ) -> torch.Tensor:
        """Return outcome predictions for covariates x and treatment t."""
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
        elif t.dim() == 0:
            t = t.expand(x.size(0)).to(torch.long)
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))


__all__ = ["OpenMatch"]
```

### 3.5 Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `tau` | 0.95 | FixMatch confidence threshold |
| `tau_ova` | 0.5 | OVA threshold for OOD detection |
| `lambda_socr` | 0.5 | Weight for SOCR loss |
| `lambda_u` | 1.0 | Weight for unsupervised FixMatch loss |

---

## 4. UASD: Uncertainty-Aware Self-Distillation

**Citation**: Chen, Y., Zhu, X., Li, W., & Gong, S. (2020). Semi-Supervised Learning under Class Distribution Mismatch. *AAAI 2020*. [6]

### 4.1 Core Idea

Replace hard pseudo-labels with **soft targets** that preserve predictive uncertainty. Use accumulated confidence scores to filter OOD samples on-the-fly.

### 4.2 Mathematical Formulation

For a sample $x$, the soft target is:

$$q(x) = \frac{1}{T} \sum_{t=1}^{T} p_{\theta^{(t)}}(x)$$

where $p_{\theta^{(t)}}$ is the model's prediction at training step $t$. The OOD score is derived from the entropy of $q(x)$:

$$s_{\text{OOD}}(x) = -\sum_k q_k(x) \log q_k(x)$$

Samples with high entropy (low confidence) are down-weighted or filtered.

### 4.3 Implementation for XTYLearner

```python
# xtylearner/models/uasd.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model
from .utils import ramp_up_sigmoid


@register_model("uasd")
class UASD(nn.Module):
    """Uncertainty-Aware Self-Distillation for Open-Set SSL.

    Uses soft targets with EMA accumulation to detect and filter OOD samples.
    Preserves predictive uncertainty to avoid catastrophic error propagation.

    References
    ----------
    Chen et al., "Semi-Supervised Learning under Class Distribution Mismatch",
    AAAI 2020.
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int = 2,
        *,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: callable | None = None,
        residual: bool = False,
        # UASD params
        temperature: float = 2.0,
        ema_decay: float = 0.999,
        entropy_threshold: float = 0.5,
        lambda_u: float = 1.0,
        ramp_up: int = 40,
        # Soft target storage
        max_unlabelled: int = 50000,
    ) -> None:
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.ema_decay = ema_decay
        self.entropy_threshold = entropy_threshold
        self.lambda_u = lambda_u
        self.ramp_up = ramp_up
        self.step_count = 0

        # Outcome network
        self.outcome = make_mlp(
            [d_x + k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )

        # Treatment classifier
        self.classifier = make_mlp(
            [d_x + d_y, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )

        # EMA soft targets for unlabelled samples (indexed by position)
        self.register_buffer(
            "soft_targets", torch.zeros(max_unlabelled, k)
        )
        self.register_buffer(
            "target_counts", torch.zeros(max_unlabelled)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict outcome y from covariates x and treatment t."""
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))

    def _get_soft_probs(
        self, logits: torch.Tensor, temperature: float | None = None
    ) -> torch.Tensor:
        """Get temperature-scaled softmax probabilities."""
        T = temperature or self.temperature
        return F.softmax(logits / T, dim=-1)

    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution."""
        log_probs = torch.log(probs.clamp(min=1e-8))
        return -(probs * log_probs).sum(dim=-1)

    def _normalized_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Entropy normalized to [0, 1] by max entropy."""
        ent = self._entropy(probs)
        max_ent = torch.log(torch.tensor(self.k, dtype=probs.dtype, device=probs.device))
        return ent / max_ent

    def _update_soft_targets(
        self,
        indices: torch.Tensor,
        probs: torch.Tensor,
    ) -> None:
        """Update EMA soft targets for given sample indices."""
        alpha = self.ema_decay
        for i, idx in enumerate(indices):
            idx = idx.item()
            if self.target_counts[idx] == 0:
                self.soft_targets[idx] = probs[i]
            else:
                self.soft_targets[idx] = (
                    alpha * self.soft_targets[idx] + (1 - alpha) * probs[i]
                )
            self.target_counts[idx] += 1

    def _get_accumulated_targets(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get accumulated soft targets and their entropy scores."""
        targets = self.soft_targets[indices]
        # Normalize to ensure valid probability distribution
        targets = targets / targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        entropy = self._normalized_entropy(targets)
        return targets, entropy

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_obs: torch.Tensor,
        unlabelled_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute UASD loss with soft distillation and OOD filtering.

        Parameters
        ----------
        unlabelled_indices:
            Global indices of unlabelled samples (for soft target tracking).
            If None, uses sequential indices based on batch position.
        """
        labelled = t_obs >= 0
        loss = torch.tensor(0.0, device=x.device)

        inp = torch.cat([x, y], dim=-1)
        logits = self.classifier(inp)

        # === Supervised losses ===
        if labelled.any():
            t_use = t_obs[labelled].clamp_min(0)
            t_onehot = F.one_hot(t_use.to(torch.long), self.k).float()

            # Outcome loss
            y_hat = self.outcome(torch.cat([x[labelled], t_onehot], dim=-1))
            loss += F.mse_loss(y_hat, y[labelled])

            # Classification loss
            loss += F.cross_entropy(logits[labelled], t_use)

        # === Unsupervised losses with soft distillation ===
        if (~labelled).any():
            logits_u = logits[~labelled]
            probs_u = self._get_soft_probs(logits_u)

            # Get or create indices for unlabelled samples
            if unlabelled_indices is None:
                # Use sequential indices (assumes consistent batching)
                n_unlabelled = (~labelled).sum().item()
                unlabelled_indices = torch.arange(
                    n_unlabelled, device=x.device
                )

            # Update soft targets with current predictions
            with torch.no_grad():
                self._update_soft_targets(unlabelled_indices, probs_u.detach())

            # Get accumulated soft targets
            soft_targets, entropy = self._get_accumulated_targets(unlabelled_indices)

            # Filter based on entropy (low entropy = confident = likely in-dist)
            # Samples with entropy > threshold are considered OOD
            ood_weight = torch.sigmoid(
                10.0 * (self.entropy_threshold - entropy)
            )  # Soft filtering

            # KL divergence to soft targets (temperature-scaled)
            log_probs = F.log_softmax(logits_u / self.temperature, dim=-1)
            kl_loss = F.kl_div(
                log_probs,
                soft_targets.detach(),
                reduction='none'
            ).sum(dim=-1)

            # Weight by inverse OOD score
            L_distill = (ood_weight * kl_loss).mean()

            lam = ramp_up_sigmoid(self.step_count, self.ramp_up, self.lambda_u)
            loss += lam * L_distill

        return loss

    def step(self) -> None:
        """Increment step counter after optimizer step."""
        self.step_count += 1

    @torch.no_grad()
    def predict_treatment_proba(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Predict treatment probabilities."""
        inp = torch.cat([x, y], dim=-1)
        logits = self.classifier(inp)
        return logits.softmax(dim=-1)

    @torch.no_grad()
    def predict_ood_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return OOD score based on prediction entropy."""
        probs = self.predict_treatment_proba(x, y)
        return self._normalized_entropy(probs)

    @torch.no_grad()
    def predict_outcome(
        self, x: torch.Tensor, t: int | torch.Tensor
    ) -> torch.Tensor:
        """Return outcome predictions for covariates x and treatment t."""
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
        elif t.dim() == 0:
            t = t.expand(x.size(0)).to(torch.long)
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))


__all__ = ["UASD"]
```

### 4.4 Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `temperature` | 2.0 | Softmax temperature for soft targets |
| `ema_decay` | 0.999 | EMA decay for soft target accumulation |
| `entropy_threshold` | 0.5 | Normalized entropy threshold for OOD |
| `max_unlabelled` | 50000 | Max samples to track soft targets for |

---

## 5. DS³L: Distribution-Aware Self-Training

**Citation**: Guo, L., & Li, Z. (2020). DS³L: Distribution-Aware Self-Training for Semi-supervised Learning with Out-of-Distribution Data. [7]

### 5.1 Core Idea

Learn a **meta-reweighting network** that assigns weights to unlabelled samples based on their estimated benefit to the learning task. Unlike UASD, DS³L uses a held-out validation set to guide the weighting.

### 5.2 Key Insight

The meta-reweighter learns to identify which OOD samples are actually harmful vs. potentially beneficial. Some OOD data can provide useful regularization—complete filtering is suboptimal.

### 5.3 Simplified Implementation

For XTYLearner, we recommend a simplified version that uses gradient-based meta-learning:

```python
# Pseudocode for DS³L-style meta-weighting
def compute_sample_weights(model, x_u, x_val, y_val):
    """Compute weights via one-step gradient on validation."""
    # Forward on validation
    val_loss = model.supervised_loss(x_val, y_val)

    # Get gradients w.r.t. model params
    val_grads = torch.autograd.grad(val_loss, model.parameters())

    # For each unlabelled sample, compute influence
    weights = []
    for x_i in x_u:
        # Approximate influence via gradient alignment
        u_loss = model.unsupervised_loss(x_i)
        u_grads = torch.autograd.grad(u_loss, model.parameters())

        # Cosine similarity: aligned gradients → beneficial sample
        alignment = cosine_similarity(val_grads, u_grads)
        weights.append(torch.relu(alignment))  # Only keep positive

    return torch.stack(weights)
```

---

## 6. Confidence-Based Soft Weighting (Minimal Approach)

> ✅ **STATUS: IMPLEMENTED** in `xtylearner/models/mean_teacher.py`

For a lightweight solution that doesn't require new model architectures, add confidence-based weighting to existing Mean Teacher or FixMatch implementations.

### 6.1 Usage

The Mean Teacher model now includes built-in OOD weighting. Enable it via constructor parameters:

```python
from xtylearner.models import get_model

model = get_model(
    "mean_teacher",
    d_x=10,
    d_y=1,
    k=3,
    # OOD weighting parameters (all optional, shown with defaults)
    ood_weighting=True,       # Enable/disable OOD weighting
    ood_threshold=0.5,        # Confidence threshold (samples below are down-weighted)
    ood_sharpness=10.0,       # Sigmoid sharpness for soft weighting
)
```

To disable OOD weighting and revert to standard Mean Teacher behavior:

```python
model = get_model("mean_teacher", d_x=10, d_y=1, k=3, ood_weighting=False)
```

### 6.2 Implementation Details

The weighting is applied per-sample in the consistency loss:

```python
def _compute_ood_weights(self, teacher_probs: torch.Tensor) -> torch.Tensor:
    """Compute per-sample OOD weights based on teacher confidence."""
    # Max confidence as in-distribution indicator
    max_conf, _ = teacher_probs.max(dim=-1)

    # Soft sigmoid weighting centered at threshold
    # High confidence -> weight ≈ 1 (likely in-distribution)
    # Low confidence -> weight ≈ 0 (likely OOD)
    weights = torch.sigmoid(
        self.ood_sharpness * (max_conf - self.ood_threshold)
    )
    return weights
```

The consistency loss then becomes:

```python
if self.ood_weighting:
    ood_weights = self._compute_ood_weights(teacher_probs)
    per_sample_mse = ((student_probs - teacher_probs) ** 2).mean(dim=-1)
    L_cons = (ood_weights * per_sample_mse).mean()
else:
    L_cons = F.mse_loss(student_probs, teacher_probs)
```

### 6.3 Diagnostics

Use `predict_ood_score()` to inspect which samples are being down-weighted:

```python
# Get OOD scores for unlabelled data (higher = more likely OOD)
ood_scores = model.predict_ood_score(x_unlabelled, y_unlabelled)

# Samples with score > 0.5 are being significantly down-weighted
likely_ood = ood_scores > 0.5
print(f"Detected {likely_ood.sum()} likely OOD samples out of {len(ood_scores)}")
```

### 6.4 Hyperparameter Tuning

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `ood_threshold` | 0.5 | Lower if many OOD samples expected; raise if false positives are an issue |
| `ood_sharpness` | 10.0 | Higher = sharper cutoff; lower = more gradual weighting |

---

## 7. Energy-Based OOD Detection

**Citation**: Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based Out-of-distribution Detection. *NeurIPS 2020*. [8]

### 7.1 Core Idea

Use the **energy score** of the softmax distribution as an OOD indicator. Energy is defined as:

$$E(x) = -T \cdot \log \sum_k \exp(f_k(x) / T)$$

where $f_k(x)$ is the k-th logit. Lower energy indicates in-distribution.

### 7.2 Implementation

```python
def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute energy score for OOD detection (lower = more in-dist)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


def energy_weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0,
    energy_margin: float = -5.0,
) -> torch.Tensor:
    """Weight samples by energy score."""
    energy = energy_score(logits, temperature)

    # Soft weight based on energy (lower energy = higher weight)
    weight = torch.sigmoid(energy_margin - energy)

    ce = F.cross_entropy(logits, targets, reduction='none')
    return (weight * ce).mean()
```

---

## 8. Clone: Closed Loop Networks

**Citation**: Closed Loop Networks for Open-Set Semi-Supervised Learning. *Information Sciences, 2024*. [9]

### 8.1 Core Idea

Use **two independent networks** instead of a shared backbone:
1. **OOD Detector**: Specialized for detecting out-of-distribution samples
2. **Classifier**: Specialized for K-way classification

A feedback loop connects them: the OOD detector filters samples for the classifier, and the classifier's predictions inform the OOD detector's training.

### 8.2 Architecture

```
┌─────────────────┐     Filter      ┌─────────────────┐
│   OOD Detector  │ ──────────────► │    Classifier   │
│   (Network A)   │                 │   (Network B)   │
└────────┬────────┘                 └────────┬────────┘
         │                                   │
         │◄──────────── Feedback ────────────┘
         │         (confident predictions
         │          inform OOD training)
```

### 8.3 Implementation Sketch

```python
@register_model("clone")
class Clone(nn.Module):
    """Closed Loop Networks for Open-Set SSL."""

    def __init__(self, d_x, d_y, k, hidden_dims=(128, 128), ...):
        super().__init__()

        # Independent OOD detection network
        self.ood_network = make_mlp([d_x + d_y, *hidden_dims, 1])

        # Independent classifier network
        self.classifier_network = make_mlp([d_x + d_y, *hidden_dims, k])

        # Outcome network (shared)
        self.outcome = make_mlp([d_x + k, *hidden_dims, d_y])

    def predict_ood(self, inp):
        """Binary OOD prediction."""
        return torch.sigmoid(self.ood_network(inp))

    def predict_class(self, inp):
        """K-way classification."""
        return self.classifier_network(inp)

    def loss(self, x, y, t_obs):
        inp = torch.cat([x, y], dim=-1)
        labelled = t_obs >= 0

        # Phase 1: Train OOD detector on labelled (known in-dist)
        if labelled.any():
            ood_pred = self.predict_ood(inp[labelled])
            # Labelled samples are in-distribution (target = 0)
            L_ood = F.binary_cross_entropy(ood_pred, torch.zeros_like(ood_pred))

        # Phase 2: Train classifier on filtered unlabelled
        if (~labelled).any():
            with torch.no_grad():
                ood_scores = self.predict_ood(inp[~labelled])
                is_inlier = ood_scores < 0.5

            if is_inlier.any():
                # FixMatch-style pseudo-labeling on inliers
                ...

        # Phase 3: Feedback - use confident classifier predictions
        # to train OOD detector on pseudo-OOD samples
        ...
```

---

## 9. MixMOOD: Ante-Hoc Dataset Selection

**Citation**: Motamed, S., et al. (2020). MixMOOD: Measuring OOD-ness of Unlabeled Data for Semi-Supervised Learning. [10]

### 9.1 Core Idea

Use **Deep Dataset Dissimilarity Measures (DeDiMs)** to rank unlabelled datasets *before training*. This is useful when you have multiple unlabelled sources and need to select the most beneficial one.

### 9.2 When to Use

- You have multiple unlabelled datasets to choose from
- You want to pre-filter a large unlabelled pool before SSL training
- Computational budget is limited and you can't afford trial-and-error

### 9.3 Implementation

```python
def compute_dedim(
    model: nn.Module,
    labelled_loader: DataLoader,
    unlabelled_loader: DataLoader,
) -> float:
    """Compute Deep Dataset Dissimilarity Measure.

    Uses Maximum Mean Discrepancy (MMD) in feature space.
    Lower DeDiM = more similar = safer to use for SSL.
    """
    model.eval()

    # Extract features from labelled data
    lab_features = []
    for x, _ in labelled_loader:
        with torch.no_grad():
            feat = model.get_features(x)
            lab_features.append(feat)
    lab_features = torch.cat(lab_features)

    # Extract features from unlabelled data
    unlab_features = []
    for x, in unlabelled_loader:
        with torch.no_grad():
            feat = model.get_features(x)
            unlab_features.append(feat)
    unlab_features = torch.cat(unlab_features)

    # Compute MMD
    from .utils import mmd
    return mmd(lab_features, unlab_features).item()
```

---

## 10. Comparative Summary

| Method | OOD Detection | Collapse Prevention | Complexity | Best For |
|--------|--------------|---------------------|------------|----------|
| **OpenMatch** | OVA classifier | SOCR consistency | Medium | General OSSL |
| **UASD** | Entropy of soft targets | Soft distillation | Low | Resource-limited |
| **DS³L** | Meta-reweighting | Gradient alignment | High | When validation available |
| **Confidence Weighting** | Max probability | Soft weighting | Minimal | Quick integration |
| **Energy-Based** | Energy score | Margin weighting | Low | Well-calibrated models |
| **Clone** | Dedicated network | Decoupled training | High | Severe OOD mismatch |
| **MixMOOD** | DeDiM ranking | Pre-filtering | Low | Dataset selection |

---

## 11. Recommended Implementation Order

1. ✅ **COMPLETE**: Confidence-based weighting added to Mean Teacher (Section 6)
   - Implemented in `xtylearner/models/mean_teacher.py`
   - New parameters: `ood_weighting`, `ood_threshold`, `ood_sharpness`
   - New method: `predict_ood_score()` for diagnostics
2. **If insufficient**: Implement UASD for soft distillation (Section 4)
3. **For best results**: Implement OpenMatch with OVA + SOCR (Section 3)
4. **For severe mismatch**: Consider Clone architecture (Section 8)

---

## 12. References

[1] Arazo, E., Ortego, D., Albert, P., O'Connor, N. E., & McGuinness, K. (2020). Pseudo-labeling and confirmation bias in deep semi-supervised learning. *IJCNN 2020*.

[2] Zhao, Z., et al. (2022). Context-guided entropy minimization for semi-supervised domain adaptation. *Neural Networks*.

[3] Chen, Y., et al. (2023). On pseudo-labeling for class-mismatch semi-supervised learning. *arXiv:2301.06010*.

[4] Guo, Y., et al. (2024). Robust semi-supervised learning in open environments. *Frontiers of Computer Science*.

[5] Saito, K., Kim, D., & Saenko, K. (2021). OpenMatch: Open-set consistency regularization for semi-supervised learning with outliers. *NeurIPS 2021*.

[6] Chen, Y., Zhu, X., Li, W., & Gong, S. (2020). Semi-supervised learning under class distribution mismatch. *AAAI 2020*.

[7] Guo, L., & Li, Z. (2020). DS³L: Distribution-aware self-training for semi-supervised learning with out-of-distribution data.

[8] Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based out-of-distribution detection. *NeurIPS 2020*.

[9] Closed loop networks for open-set semi-supervised learning. *Information Sciences, 2024*.

[10] Motamed, S., et al. (2020). MixMOOD: Measuring OOD-ness of unlabeled data for semi-supervised learning.

[11] Wei, T., et al. (2023). Exploration and exploitation of unlabeled data for open-set semi-supervised learning. *IJCV 2024*.

[12] Wang, Q., et al. (2024). WiseOpen: Robust semi-supervised learning by wisely leveraging open-set data. *arXiv:2405.06979*.

---

## 13. Testing Strategy

When implementing these methods, validate with:

1. **Synthetic OOD injection**: Add samples with novel treatment labels to unlabelled data
2. **Varying OOD ratio**: Test with 0%, 25%, 50%, 75% OOD in unlabelled data
3. **Monitor for collapse**: Track prediction entropy over training
4. **Compare to supervised baseline**: OSSL should never perform worse

```python
def validate_ossl_method(model, X_lab, y_lab, X_unlab_clean, X_unlab_ood):
    """Validate OSSL robustness to OOD contamination."""
    results = {}

    for ood_ratio in [0.0, 0.25, 0.5, 0.75]:
        n_ood = int(len(X_unlab_clean) * ood_ratio / (1 - ood_ratio))
        X_unlab = np.vstack([X_unlab_clean, X_unlab_ood[:n_ood]])

        model.fit(X_lab, y_lab, X_unlab)
        results[ood_ratio] = model.evaluate(X_test, y_test)

    return results
```

---

*Document version: 1.0 | Last updated: 2026-01-09*
