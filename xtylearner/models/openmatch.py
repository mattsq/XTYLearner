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
