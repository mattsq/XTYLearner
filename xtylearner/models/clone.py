"""Clone: Closed Loop Networks for Open-Set SSL."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model
from .utils import ramp_up_sigmoid


@register_model("clone")
class Clone(nn.Module):
    """Closed Loop Networks for Open-Set Semi-Supervised Learning.

    Uses two independent networks with a feedback loop:
    1. OOD Detector: Specialized for detecting out-of-distribution samples
    2. Classifier: Specialized for K-way classification

    The OOD detector filters samples for the classifier, and the classifier's
    confident predictions provide feedback to improve the OOD detector.

    This decoupled architecture prevents catastrophic failure modes where a
    shared backbone causes the classifier to degrade when exposed to OOD data.

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
    tau : float
        FixMatch confidence threshold for pseudo-labeling.
    tau_ood : float
        OOD detection threshold (samples with OOD score > tau_ood are filtered).
    lambda_u : float
        Maximum weight for unsupervised classification loss.
    lambda_feedback : float
        Weight for feedback loss (classifier -> OOD detector).
    ramp_up : int
        Number of steps for unsupervised loss ramp-up.

    References
    ----------
    "Closed Loop Networks for Open-Set Semi-Supervised Learning",
    Information Sciences, 2024.
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
        tau_ood: float = 0.5,
        lambda_u: float = 1.0,
        lambda_feedback: float = 0.5,
        # Ramp-up
        ramp_up: int = 40,
    ) -> None:
        super().__init__()
        self.k = k
        self.d_x = d_x
        self.d_y = d_y
        self.tau = tau
        self.tau_ood = tau_ood
        self.lambda_u = lambda_u
        self.lambda_feedback = lambda_feedback
        self.ramp_up = ramp_up
        self.step_count = 0

        # Outcome network (shared)
        self.outcome = make_mlp(
            [d_x + k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )

        # Independent OOD detection network
        # Takes (x, y) -> binary OOD score
        self.ood_network = make_mlp(
            [d_x + d_y, *hidden_dims, 1],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )

        # Independent classifier network
        # Takes (x, y) -> K-way treatment logits
        self.classifier_network = make_mlp(
            [d_x + d_y, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict outcome y from covariates x and treatment t."""
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))

    def _predict_ood_prob(self, inp: torch.Tensor) -> torch.Tensor:
        """Predict probability of being OOD.

        Returns a value in [0, 1] where higher means more likely OOD.
        """
        return torch.sigmoid(self.ood_network(inp))

    def _get_class_logits(self, inp: torch.Tensor) -> torch.Tensor:
        """Get K-way classification logits."""
        return self.classifier_network(inp)

    def _add_noise(self, x: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
        """Add Gaussian noise for data augmentation."""
        if self.training:
            return x + scale * torch.randn_like(x)
        return x

    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute Clone loss with decoupled OOD detection and classification.

        Training proceeds in three phases:
        1. Train OOD detector on labelled data (known in-distribution)
        2. Filter unlabelled data and train classifier with pseudo-labels
        3. Use confident predictions to provide feedback to OOD detector

        Parameters
        ----------
        x : torch.Tensor
            Covariates [batch_size, d_x].
        y : torch.Tensor
            Outcomes [batch_size, d_y].
        t_obs : torch.Tensor
            Observed treatments [batch_size], with -1 for unlabelled samples.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        labelled = t_obs >= 0
        loss = torch.tensor(0.0, device=x.device)

        inp = torch.cat([x, y], dim=-1)

        # === Phase 1: Supervised training on labelled data ===
        if labelled.any():
            inp_lab = inp[labelled]
            t_use = t_obs[labelled].clamp_min(0)
            t_onehot = F.one_hot(t_use.to(torch.long), self.k).float()

            # Outcome loss
            y_hat = self.outcome(torch.cat([x[labelled], t_onehot], dim=-1))
            loss = loss + F.mse_loss(y_hat, y[labelled])

            # Classification loss
            logits_lab = self._get_class_logits(inp_lab)
            loss = loss + F.cross_entropy(logits_lab, t_use)

            # OOD detector loss: labelled samples are known in-distribution
            # Target = 0 (not OOD)
            ood_pred_lab = self._predict_ood_prob(inp_lab)
            target_lab = torch.zeros_like(ood_pred_lab)
            loss = loss + F.binary_cross_entropy(ood_pred_lab, target_lab)

        # === Phase 2 & 3: Unsupervised training on unlabelled data ===
        if (~labelled).any():
            inp_u = inp[~labelled]

            # Filter OOD samples using the OOD detector
            with torch.no_grad():
                ood_probs = self._predict_ood_prob(inp_u)
                ood_scores = ood_probs.squeeze(-1)
                is_inlier = ood_scores < self.tau_ood

            # Phase 2: FixMatch-style pseudo-labeling on filtered inliers
            if is_inlier.any():
                inp_inlier = inp_u[is_inlier]

                # Weak augmentation -> generate pseudo-labels
                with torch.no_grad():
                    inp_weak = self._add_noise(inp_inlier, scale=0.1)
                    logits_weak = self._get_class_logits(inp_weak)
                    probs = logits_weak.softmax(dim=-1)
                    max_prob, pseudo_label = probs.max(dim=-1)
                    # Only use confident predictions
                    conf_mask = max_prob >= self.tau

                # Strong augmentation -> classification loss
                if conf_mask.any():
                    inp_strong = self._add_noise(inp_inlier[conf_mask], scale=0.2)
                    logits_strong = self._get_class_logits(inp_strong)
                    L_unsup = F.cross_entropy(logits_strong, pseudo_label[conf_mask])
                    lam = ramp_up_sigmoid(self.step_count, self.ramp_up, self.lambda_u)
                    loss = loss + lam * L_unsup

                    # Phase 3: Feedback - use confident predictions to train OOD detector
                    # Confident inlier predictions should be marked as in-distribution
                    with torch.no_grad():
                        # Create feedback targets based on confidence
                        # Very confident predictions -> strong in-dist signal (target = 0)
                        # We use the confidence as a soft signal
                        feedback_weight = max_prob[conf_mask]

                    ood_pred_feedback = self._predict_ood_prob(inp_inlier[conf_mask])
                    # Target: confident predictions should be classified as in-dist (0)
                    target_feedback = torch.zeros_like(ood_pred_feedback)
                    # Weight by confidence - more confident = stronger signal
                    L_feedback = (
                        feedback_weight.unsqueeze(-1)
                        * F.binary_cross_entropy(
                            ood_pred_feedback, target_feedback, reduction="none"
                        )
                    ).mean()
                    loss = loss + self.lambda_feedback * L_feedback

            # Additional: Use detected OOD samples to improve OOD detector
            # Samples with high OOD scores should be reinforced as OOD
            if (~is_inlier).any():
                inp_ood = inp_u[~is_inlier]
                ood_pred_ood = self._predict_ood_prob(inp_ood)
                # Target = 1 (is OOD)
                target_ood = torch.ones_like(ood_pred_ood)
                # Soft weighting based on how far beyond threshold
                with torch.no_grad():
                    ood_weight = torch.sigmoid(
                        10.0 * (ood_scores[~is_inlier].unsqueeze(-1) - self.tau_ood)
                    )
                L_ood_reinforce = (
                    ood_weight * F.binary_cross_entropy(
                        ood_pred_ood, target_ood, reduction="none"
                    )
                ).mean()
                loss = loss + 0.5 * L_ood_reinforce

        return loss

    def step(self) -> None:
        """Increment step counter after optimizer step."""
        self.step_count += 1

    @torch.no_grad()
    def predict_treatment_proba(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Predict treatment probabilities using the classifier network.

        Parameters
        ----------
        x : torch.Tensor
            Covariates [batch_size, d_x].
        y : torch.Tensor
            Outcomes [batch_size, d_y].

        Returns
        -------
        torch.Tensor
            Treatment probabilities [batch_size, k].
        """
        inp = torch.cat([x, y], dim=-1)
        logits = self._get_class_logits(inp)
        return logits.softmax(dim=-1)

    @torch.no_grad()
    def predict_ood_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return OOD score: higher = more likely OOD.

        Uses the dedicated OOD detection network.

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
        inp = torch.cat([x, y], dim=-1)
        return self._predict_ood_prob(inp).squeeze(-1)

    @torch.no_grad()
    def predict_outcome(
        self, x: torch.Tensor, t: int | torch.Tensor
    ) -> torch.Tensor:
        """Return outcome predictions for covariates x and treatment t.

        Parameters
        ----------
        x : torch.Tensor
            Covariates [batch_size, d_x].
        t : int | torch.Tensor
            Treatment assignment (single int or tensor of shape [batch_size]).

        Returns
        -------
        torch.Tensor
            Predicted outcomes [batch_size, d_y].
        """
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
        elif t.dim() == 0:
            t = t.expand(x.size(0)).to(torch.long)
        t_onehot = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome(torch.cat([x, t_onehot], dim=-1))


__all__ = ["Clone"]
