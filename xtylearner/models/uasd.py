"""Uncertainty-Aware Self-Distillation for Open-Set SSL."""

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

    The key insight is that OOD samples tend to have high-entropy predictions
    over time. By tracking the EMA of predictions for each unlabelled sample,
    we can identify and down-weight OOD samples based on their accumulated
    uncertainty.

    Parameters
    ----------
    d_x : int
        Dimension of covariates.
    d_y : int
        Dimension of outcome.
    k : int
        Number of treatment classes.
    hidden_dims : tuple[int, ...]
        Hidden layer dimensions for networks.
    activation : type[nn.Module]
        Activation function class.
    dropout : float | None
        Dropout probability.
    norm_layer : callable | None
        Normalization layer constructor.
    residual : bool
        Whether to use residual connections.
    temperature : float
        Softmax temperature for soft targets (higher = softer).
    ema_decay : float
        EMA decay rate for soft target accumulation.
    entropy_threshold : float
        Normalized entropy threshold for OOD detection (0-1 scale).
        Samples with entropy above this are considered OOD.
    lambda_u : float
        Maximum weight for unsupervised distillation loss.
    ramp_up : int
        Number of steps for consistency loss ramp-up.
    max_unlabelled : int
        Maximum number of unlabelled samples to track soft targets for.

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
        self.d_x = d_x
        self.d_y = d_y
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
        self.register_buffer("soft_targets", torch.zeros(max_unlabelled, k))
        self.register_buffer("target_counts", torch.zeros(max_unlabelled))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict outcome y from covariates x and treatment t."""
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))

    def _get_soft_probs(
        self, logits: torch.Tensor, temperature: float | None = None
    ) -> torch.Tensor:
        """Get temperature-scaled softmax probabilities."""
        T = temperature if temperature is not None else self.temperature
        return F.softmax(logits / T, dim=-1)

    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution."""
        log_probs = torch.log(probs.clamp(min=1e-8))
        return -(probs * log_probs).sum(dim=-1)

    def _normalized_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Entropy normalized to [0, 1] by max entropy."""
        ent = self._entropy(probs)
        max_ent = torch.log(
            torch.tensor(self.k, dtype=probs.dtype, device=probs.device)
        )
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
        x : torch.Tensor
            Covariates [batch_size, d_x].
        y : torch.Tensor
            Outcomes [batch_size, d_y].
        t_obs : torch.Tensor
            Observed treatments [batch_size], with -1 for unlabelled samples.
        unlabelled_indices : torch.Tensor | None
            Global indices of unlabelled samples (for soft target tracking).
            If None, uses sequential indices based on batch position.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
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
            loss = loss + F.mse_loss(y_hat, y[labelled])

            # Classification loss
            loss = loss + F.cross_entropy(logits[labelled], t_use)

        # === Unsupervised losses with soft distillation ===
        if (~labelled).any():
            logits_u = logits[~labelled]
            probs_u = self._get_soft_probs(logits_u)

            # Get or create indices for unlabelled samples
            if unlabelled_indices is None:
                # Use sequential indices (assumes consistent batching)
                n_unlabelled = (~labelled).sum().item()
                unlabelled_indices = torch.arange(n_unlabelled, device=x.device)

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
                log_probs, soft_targets.detach(), reduction="none"
            ).sum(dim=-1)

            # Weight by inverse OOD score
            L_distill = (ood_weight * kl_loss).mean()

            lam = ramp_up_sigmoid(self.step_count, self.ramp_up, self.lambda_u)
            loss = loss + lam * L_distill

        return loss

    def step(self) -> None:
        """Increment step counter after optimizer step."""
        self.step_count += 1

    def reset_soft_targets(self) -> None:
        """Reset accumulated soft targets (e.g., at start of new epoch)."""
        self.soft_targets.zero_()
        self.target_counts.zero_()

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
        """Return OOD score based on prediction entropy.

        Higher scores indicate samples that are more likely OOD.
        Uses normalized entropy of model predictions.

        Parameters
        ----------
        x : torch.Tensor
            Covariates [batch_size, d_x].
        y : torch.Tensor
            Outcomes [batch_size, d_y].

        Returns
        -------
        torch.Tensor
            OOD scores in [0, 1], shape [batch_size]. Higher = more likely OOD.
        """
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
