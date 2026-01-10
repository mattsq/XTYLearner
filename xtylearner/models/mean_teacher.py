from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model
from .utils import ramp_up_sigmoid


@register_model("mean_teacher")
class MeanTeacher(nn.Module):
    """Mean Teacher model for semi-supervised treatment inference.

    Implementation follows the canonical Mean Teacher approach with:
    - Dual heads for student (classification vs consistency)
    - Feature-wise noise calibration using running statistics
    - Gaussian ramp-up for consistency weight
    - External EMA update after optimizer step
    - **Confidence-based OOD weighting** to handle unlabelled data with
      out-of-distribution samples (treatments not in labelled set)

    The OOD weighting uses teacher prediction confidence as an in-distribution
    indicator: low confidence samples are down-weighted in the consistency loss
    to prevent confirmation bias and model collapse from OOD samples.

    References
    ----------
    - Tarvainen & Valpola, "Mean teachers are better role models", NeurIPS 2017
    - Saito et al., "OpenMatch: Open-set Consistency Regularization", NeurIPS 2021
    - Chen et al., "Semi-Supervised Learning under Class Distribution Mismatch", AAAI 2020
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
        ema_decay: float = 0.99,
        ramp_up: int = 40,
        cons_max: float = 1.0,
        noise_scale: float = 0.1,
        noise_momentum: float = 0.01,
        logit_distance_cost: float = 0.01,
        # OOD weighting parameters
        ood_weighting: bool = True,
        ood_threshold: float = 0.5,
        ood_sharpness: float = 10.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.d_x = d_x
        self.d_y = d_y

        # Outcome prediction network
        self.outcome = make_mlp(
            [d_x + k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )

        # Student network: backbone + dual heads
        self.student_backbone = make_mlp(
            [d_x + d_y, *hidden_dims],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )
        self.student_class_head = nn.Linear(hidden_dims[-1], k)  # For supervised loss
        self.student_cons_head = nn.Linear(hidden_dims[-1], k)   # For consistency loss

        # Teacher network: backbone + consistency head (EMA copy)
        self.teacher_backbone = deepcopy(self.student_backbone)
        self.teacher_head = deepcopy(self.student_cons_head)
        for p in self.teacher_backbone.parameters():
            p.requires_grad_(False)
        for p in self.teacher_head.parameters():
            p.requires_grad_(False)

        # Hyperparameters
        self.ema_decay = ema_decay
        self.ramp_up = ramp_up
        self.cons_max = cons_max
        self.noise_scale = noise_scale
        self.noise_momentum = noise_momentum
        self.logit_distance_cost = logit_distance_cost
        self.step_count = 0  # Renamed to avoid shadowing step() method

        # OOD weighting parameters
        self.ood_weighting = ood_weighting
        self.ood_threshold = ood_threshold
        self.ood_sharpness = ood_sharpness

        # Running statistics for feature-wise noise calibration
        self.register_buffer("running_x_std", torch.ones(d_x))
        self.register_buffer("running_y_std", torch.ones(d_y))

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict outcome ``y`` from covariates ``x`` and treatment ``t``."""
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))

    # --------------------------------------------------------------
    def _consistency_weight(self) -> float:
        """Get current consistency loss weight using Gaussian ramp-up."""
        return ramp_up_sigmoid(self.step_count, self.ramp_up, self.cons_max)

    # --------------------------------------------------------------
    def _update_running_stats(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Update running std estimates with current batch."""
        if self.training:
            with torch.no_grad():
                batch_x_std = x.std(0).clamp(min=1e-6)
                batch_y_std = y.std(0).clamp(min=1e-6)
                self.running_x_std.lerp_(batch_x_std, self.noise_momentum)
                self.running_y_std.lerp_(batch_y_std, self.noise_momentum)

    # --------------------------------------------------------------
    def _add_noise(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add calibrated Gaussian noise to inputs using running statistics."""
        if not self.training:
            return x, y
        x_noisy = x + self.noise_scale * self.running_x_std * torch.randn_like(x)
        y_noisy = y + self.noise_scale * self.running_y_std * torch.randn_like(y)
        return x_noisy, y_noisy

    # --------------------------------------------------------------
    def _student_forward(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through student network.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (class_logits, cons_logits) for supervised and consistency losses
        """
        features = self.student_backbone(inp)
        return self.student_class_head(features), self.student_cons_head(features)

    # --------------------------------------------------------------
    def _teacher_forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass through teacher network.

        Returns
        -------
        torch.Tensor
            Consistency logits for mean teacher target
        """
        features = self.teacher_backbone(inp)
        return self.teacher_head(features)

    # --------------------------------------------------------------
    def _compute_ood_weights(self, teacher_probs: torch.Tensor) -> torch.Tensor:
        """Compute per-sample OOD weights based on teacher confidence.

        Samples with low max probability (uncertain predictions) are likely
        out-of-distribution and receive lower weight in the consistency loss.
        This prevents confirmation bias from propagating incorrect pseudo-labels.

        Parameters
        ----------
        teacher_probs:
            Softmax probabilities from teacher network [batch_size, k]

        Returns
        -------
        torch.Tensor
            Per-sample weights in [0, 1], shape [batch_size]
        """
        # Max confidence as in-distribution indicator
        max_conf, _ = teacher_probs.max(dim=-1)

        # Soft sigmoid weighting centered at threshold
        # High confidence -> weight ≈ 1 (likely in-distribution)
        # Low confidence -> weight ≈ 0 (likely OOD)
        weights = torch.sigmoid(
            self.ood_sharpness * (max_conf - self.ood_threshold)
        )
        return weights

    # --------------------------------------------------------------
    def update_teacher(self) -> None:
        """Update teacher network using EMA of student weights.

        IMPORTANT: Call this AFTER optimizer.step() so teacher sees
        the most recent student weights.
        """
        with torch.no_grad():
            alpha = self.ema_decay
            # Update backbone
            for p_s, p_t in zip(
                self.student_backbone.parameters(),
                self.teacher_backbone.parameters()
            ):
                p_t.data.mul_(alpha).add_(p_s.data, alpha=1 - alpha)
            # Update consistency head only (not class head)
            for p_s, p_t in zip(
                self.student_cons_head.parameters(),
                self.teacher_head.parameters()
            ):
                p_t.data.mul_(alpha).add_(p_s.data, alpha=1 - alpha)

    # --------------------------------------------------------------
    def step(self) -> None:
        """Call after optimizer.step() to update teacher and increment counter."""
        self.update_teacher()
        self.step_count += 1

    # --------------------------------------------------------------
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute semi-supervised loss.

        Parameters
        ----------
        x:
            Covariates [batch_size, d_x]
        y:
            Outcomes [batch_size, d_y]
        t_obs:
            Observed treatments [batch_size], with -1 for unlabeled samples

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        # Update running statistics for noise calibration
        self._update_running_stats(x, y)

        # Add independent noise for student and teacher inputs
        x_s, y_s = self._add_noise(x, y)
        x_t, y_t = self._add_noise(x, y)

        inp_s = torch.cat([x_s, y_s], dim=-1)
        inp_t = torch.cat([x_t, y_t], dim=-1)

        # Student forward (dual heads)
        class_logits, cons_logits = self._student_forward(inp_s)

        # Teacher forward (consistency target)
        with torch.no_grad():
            teacher_logits = self._teacher_forward(inp_t)

        labelled = t_obs >= 0
        loss = torch.tensor(0.0, device=x.device)

        # Supervised losses (on labeled data only)
        if labelled.any():
            t_use = t_obs[labelled].clamp_min(0)
            t_onehot = F.one_hot(t_use.to(torch.long), self.k).float()

            # Outcome prediction loss
            y_hat = self.outcome(torch.cat([x[labelled], t_onehot], dim=-1))
            loss += F.mse_loss(y_hat, y[labelled])

            # Treatment classification loss (uses class head)
            loss += F.cross_entropy(class_logits[labelled], t_use)

        # Consistency loss (on all data, uses cons head)
        student_probs = cons_logits.softmax(dim=-1)
        teacher_probs = teacher_logits.softmax(dim=-1)

        if self.ood_weighting:
            # Per-sample weighted consistency loss for OOD robustness
            # Low-confidence samples (likely OOD) receive lower weight
            ood_weights = self._compute_ood_weights(teacher_probs)
            per_sample_mse = ((student_probs - teacher_probs) ** 2).mean(dim=-1)
            L_cons = (ood_weights * per_sample_mse).mean()
        else:
            # Standard unweighted consistency loss
            L_cons = F.mse_loss(student_probs, teacher_probs)

        lam = self._consistency_weight()
        loss = loss + lam * L_cons

        # Logit distance cost: encourage dual heads to agree
        if self.logit_distance_cost > 0:
            L_dist = F.mse_loss(class_logits, cons_logits)
            loss = loss + self.logit_distance_cost * L_dist

        return loss

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        """Return outcome predictions for covariates ``x`` and treatment ``t``."""
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
        elif t.dim() == 0:
            t = t.expand(x.size(0)).to(torch.long)
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict treatment probabilities using teacher network (more stable)."""
        inp = torch.cat([x, y], dim=-1)
        logits = self._teacher_forward(inp)
        return logits.softmax(dim=-1)

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_ood_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict out-of-distribution score for each sample.

        Higher scores indicate samples that are likely OOD (low teacher confidence).
        This can be used for diagnostics or post-hoc filtering.

        Parameters
        ----------
        x:
            Covariates [batch_size, d_x]
        y:
            Outcomes [batch_size, d_y]

        Returns
        -------
        torch.Tensor
            OOD scores in [0, 1], shape [batch_size]. Higher = more likely OOD.
        """
        probs = self.predict_treatment_proba(x, y)
        max_conf, _ = probs.max(dim=-1)
        # Invert: low confidence = high OOD score
        return 1.0 - max_conf


__all__ = ["MeanTeacher"]
