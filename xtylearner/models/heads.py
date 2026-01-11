import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankDiagHead(nn.Module):
    """Predicts a multivariate Gaussian with low-rank plus diagonal covariance."""

    def __init__(self, in_dim: int, d_y: int, rank: int = 4) -> None:
        super().__init__()
        self.d_y, self.rank = d_y, rank
        self.mu = nn.Linear(in_dim, d_y)
        self.factor = nn.Linear(in_dim, d_y * rank)
        self.log_sd = nn.Linear(in_dim, d_y)

    def forward(self, h: torch.Tensor):
        mu = self.mu(h)
        Fmat = self.factor(h).view(-1, self.d_y, self.rank)
        sigma2 = F.softplus(self.log_sd(h)) ** 2 + 1e-6
        return mu, Fmat, sigma2


# ---------------------------------------------------------------------------
# Ordinal Classification Heads
# ---------------------------------------------------------------------------


class CumulativeLinkHead(nn.Module):
    """Ordinal prediction head using cumulative link (proportional odds) model.

    Models cumulative probabilities P(T <= j | X) = sigmoid(threshold_j - f(X))
    where thresholds are ordered: threshold_0 < threshold_1 < ... < threshold_{k-2}.

    Parameters
    ----------
    in_features
        Dimension of input features.
    k
        Number of ordinal classes.
    link
        Link function to use. Currently only ``"logit"`` (sigmoid) is supported.

    Attributes
    ----------
    fc
        Linear layer mapping input features to a single score.
    thresholds_raw
        Unconstrained parameters that are transformed to ordered thresholds.
    """

    def __init__(self, in_features: int, k: int, link: str = "logit") -> None:
        super().__init__()
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        self.k = k
        self.link = link
        self.fc = nn.Linear(in_features, 1)
        # Initialize thresholds as evenly spaced
        # Use unconstrained parameterization: threshold_j = threshold_0 + sum(softplus(deltas[:j]))
        self.threshold_base = nn.Parameter(torch.tensor(0.0))
        self.threshold_deltas = nn.Parameter(torch.zeros(k - 2))

    def _get_thresholds(self) -> torch.Tensor:
        """Compute ordered thresholds from unconstrained parameters."""
        if self.k == 2:
            return self.threshold_base.unsqueeze(0)
        deltas = F.softplus(self.threshold_deltas) + 1e-4  # Ensure positive gaps
        cumulative_deltas = torch.cumsum(deltas, dim=0)
        thresholds = torch.cat([
            self.threshold_base.unsqueeze(0),
            self.threshold_base + cumulative_deltas
        ])
        return thresholds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cumulative probabilities P(T <= j).

        Parameters
        ----------
        x
            Input features of shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Cumulative probabilities of shape ``(batch, k-1)``.
        """
        score = self.fc(x)  # (batch, 1)
        thresholds = self._get_thresholds()  # (k-1,)
        # P(T <= j) = sigmoid(threshold_j - score)
        cumprobs = torch.sigmoid(thresholds - score)  # (batch, k-1)
        return cumprobs

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class probabilities P(T = j).

        Parameters
        ----------
        x
            Input features of shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Class probabilities of shape ``(batch, k)``.
        """
        cumprobs = self.forward(x)  # (batch, k-1)
        batch_size = cumprobs.size(0)
        zeros = torch.zeros(batch_size, 1, device=cumprobs.device, dtype=cumprobs.dtype)
        ones = torch.ones(batch_size, 1, device=cumprobs.device, dtype=cumprobs.dtype)
        cumprobs_padded = torch.cat([zeros, cumprobs, ones], dim=1)
        class_probs = cumprobs_padded[:, 1:] - cumprobs_padded[:, :-1]
        return class_probs.clamp(min=0)


class CORALHead(nn.Module):
    """CORAL (Consistent Rank Logits) head for ordinal regression.

    Uses k-1 binary classifiers with shared weights but separate biases
    to predict P(T > j).

    Parameters
    ----------
    in_features
        Dimension of input features.
    k
        Number of ordinal classes.

    References
    ----------
    Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal
    regression for neural networks with application to age estimation.
    Pattern Recognition Letters.
    """

    def __init__(self, in_features: int, k: int) -> None:
        super().__init__()
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        self.k = k
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(k - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for P(T > j).

        Parameters
        ----------
        x
            Input features of shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, k-1)`` where output[:, j] is logit(P(T > j)).
        """
        score = self.fc(x)  # (batch, 1)
        logits = score - self.biases  # (batch, k-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class probabilities P(T = j).

        Parameters
        ----------
        x
            Input features of shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Class probabilities of shape ``(batch, k)``.
        """
        logits = self.forward(x)  # (batch, k-1)
        exceed_probs = torch.sigmoid(logits)  # P(T > j)

        batch_size = logits.size(0)
        ones = torch.ones(batch_size, 1, device=logits.device, dtype=logits.dtype)
        zeros = torch.zeros(batch_size, 1, device=logits.device, dtype=logits.dtype)

        exceed_padded = torch.cat([ones, exceed_probs, zeros], dim=1)  # (batch, k+1)
        class_probs = exceed_padded[:, :-1] - exceed_padded[:, 1:]  # (batch, k)
        return class_probs.clamp(min=0)


class OrdinalHead(nn.Module):
    """Factory-style ordinal prediction head supporting multiple methods.

    This is a convenience wrapper that instantiates the appropriate ordinal
    head based on the specified method.

    Parameters
    ----------
    in_features
        Dimension of input features.
    k
        Number of ordinal classes.
    method
        Ordinal method to use:
        - ``"cumulative"``: Cumulative link model (proportional odds).
        - ``"coral"``: CORAL (Consistent Rank Logits).
        - ``"standard"``: Standard softmax classification (no ordinal structure).

    Attributes
    ----------
    method
        The ordinal method being used.
    head
        The underlying prediction head module.
    """

    def __init__(
        self,
        in_features: int,
        k: int,
        method: str = "coral",
    ) -> None:
        super().__init__()
        self.method = method
        self.k = k

        if method == "cumulative":
            self.head = CumulativeLinkHead(in_features, k)
        elif method == "coral":
            self.head = CORALHead(in_features, k)
        elif method == "standard":
            self.head = nn.Linear(in_features, k)
        else:
            raise ValueError(f"Unknown ordinal method: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ordinal head.

        Returns
        -------
        torch.Tensor
            For ``"cumulative"``: cumulative probs of shape ``(batch, k-1)``.
            For ``"coral"``: logits of shape ``(batch, k-1)``.
            For ``"standard"``: logits of shape ``(batch, k)``.
        """
        return self.head(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class probabilities P(T = j).

        Parameters
        ----------
        x
            Input features of shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Class probabilities of shape ``(batch, k)``.
        """
        if self.method == "standard":
            return F.softmax(self.head(x), dim=-1)
        else:
            return self.head.predict_proba(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict ordinal class labels.

        Parameters
        ----------
        x
            Input features of shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Predicted class labels of shape ``(batch,)``.
        """
        probs = self.predict_proba(x)
        return probs.argmax(dim=-1)


__all__ = [
    "LowRankDiagHead",
    # Ordinal heads
    "CumulativeLinkHead",
    "CORALHead",
    "OrdinalHead",
]
