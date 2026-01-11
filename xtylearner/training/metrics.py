"""Common metrics and loss functions used across trainers and models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between ``pred`` and ``target``.

    Both tensors must be broadcastable to the same shape.
    Returns the average of squared differences as a scalar tensor.
    """

    return F.mse_loss(pred, target)


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between ``pred`` and ``target``."""

    return torch.mean(torch.abs(pred - target))


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root mean squared error of ``pred`` compared with ``target``."""

    return torch.sqrt(mse_loss(pred, target))


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross entropy between ``logits`` and integer class labels ``target``."""

    return F.cross_entropy(logits, target)


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Classification accuracy comparing argmax of ``logits`` and ``target``."""

    pred = logits.argmax(dim=-1)
    return (pred == target).float().mean()


# ---------------------------------------------------------------------------
# Ordinal Classification Metrics
# ---------------------------------------------------------------------------


def ordinal_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Mean Absolute Error treating ordinal classes as integers.

    Parameters
    ----------
    predictions
        Predicted class labels of shape ``(n,)``.
    targets
        True class labels of shape ``(n,)``.

    Returns
    -------
    torch.Tensor
        Scalar MAE value.
    """
    return torch.abs(predictions.float() - targets.float()).mean()


def ordinal_rmse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Root Mean Squared Error treating ordinal classes as integers.

    Parameters
    ----------
    predictions
        Predicted class labels of shape ``(n,)``.
    targets
        True class labels of shape ``(n,)``.

    Returns
    -------
    torch.Tensor
        Scalar RMSE value.
    """
    return torch.sqrt(((predictions.float() - targets.float()) ** 2).mean())


def ordinal_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    tolerance: int = 0,
) -> torch.Tensor:
    """Accuracy with tolerance for ordinal classification.

    A prediction is considered correct if it is within ``tolerance``
    classes of the true label.

    Parameters
    ----------
    predictions
        Predicted class labels of shape ``(n,)``.
    targets
        True class labels of shape ``(n,)``.
    tolerance
        Maximum allowed difference between prediction and target.
        Default 0 means exact match (standard accuracy).

    Returns
    -------
    torch.Tensor
        Scalar accuracy value.
    """
    diff = torch.abs(predictions.long() - targets.long())
    return (diff <= tolerance).float().mean()


def adjacent_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Accuracy counting adjacent predictions as correct.

    Equivalent to ``ordinal_accuracy(predictions, targets, tolerance=1)``.

    Parameters
    ----------
    predictions
        Predicted class labels of shape ``(n,)``.
    targets
        True class labels of shape ``(n,)``.

    Returns
    -------
    torch.Tensor
        Scalar accuracy value.
    """
    return ordinal_accuracy(predictions, targets, tolerance=1)


def quadratic_weighted_kappa(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Cohen's Kappa with quadratic weights for ordinal classification.

    Quadratic weighted kappa penalizes predictions that are further from
    the true class more heavily than those that are close.

    Parameters
    ----------
    predictions
        Predicted class labels of shape ``(n,)``.
    targets
        True class labels of shape ``(n,)``.
    k
        Number of ordinal classes.

    Returns
    -------
    torch.Tensor
        Scalar kappa value in [-1, 1]. Higher is better, 1 is perfect.
    """
    predictions = predictions.long()
    targets = targets.long()
    n = predictions.size(0)

    if n == 0:
        return torch.tensor(0.0, device=predictions.device)

    # Build confusion matrix
    conf_mat = torch.zeros(k, k, device=predictions.device, dtype=torch.float)
    for i in range(n):
        conf_mat[targets[i], predictions[i]] += 1

    # Weight matrix: w[i,j] = (i-j)^2 / (k-1)^2
    indices = torch.arange(k, device=predictions.device, dtype=torch.float)
    weight_mat = ((indices.unsqueeze(0) - indices.unsqueeze(1)) ** 2) / ((k - 1) ** 2)

    # Expected matrix under independence
    hist_true = conf_mat.sum(dim=1)
    hist_pred = conf_mat.sum(dim=0)
    expected = torch.outer(hist_true, hist_pred) / n

    # Weighted observed and expected
    observed_weighted = (weight_mat * conf_mat).sum()
    expected_weighted = (weight_mat * expected).sum()

    if expected_weighted == 0:
        return torch.tensor(1.0, device=predictions.device)

    kappa = 1 - observed_weighted / expected_weighted
    return kappa


def spearman_correlation(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Spearman rank correlation coefficient.

    Measures the monotonic relationship between predictions and targets.
    For ordinal classification, this captures how well the predicted
    ordering matches the true ordering.

    Parameters
    ----------
    predictions
        Predicted class labels of shape ``(n,)``.
    targets
        True class labels of shape ``(n,)``.

    Returns
    -------
    torch.Tensor
        Scalar correlation value in [-1, 1].
    """
    predictions = predictions.float()
    targets = targets.float()
    n = predictions.size(0)

    if n < 2:
        return torch.tensor(0.0, device=predictions.device)

    def _rank(x: torch.Tensor) -> torch.Tensor:
        """Compute ranks with average tie-breaking."""
        sorted_indices = torch.argsort(x)
        ranks = torch.zeros_like(x)
        ranks[sorted_indices] = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
        return ranks

    rank_pred = _rank(predictions)
    rank_target = _rank(targets)

    # Pearson correlation on ranks
    mean_pred = rank_pred.mean()
    mean_target = rank_target.mean()

    cov = ((rank_pred - mean_pred) * (rank_target - mean_target)).sum()
    std_pred = torch.sqrt(((rank_pred - mean_pred) ** 2).sum())
    std_target = torch.sqrt(((rank_target - mean_target) ** 2).sum())

    if std_pred == 0 or std_target == 0:
        return torch.tensor(0.0, device=predictions.device)

    return cov / (std_pred * std_target)


def ordinal_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> dict[str, torch.Tensor]:
    """Compute all ordinal classification metrics.

    Parameters
    ----------
    predictions
        Predicted class labels of shape ``(n,)``.
    targets
        True class labels of shape ``(n,)``.
    k
        Number of ordinal classes.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing:
        - ``accuracy``: Exact match accuracy
        - ``adjacent_accuracy``: Off-by-one tolerance accuracy
        - ``mae``: Mean Absolute Error on class indices
        - ``rmse``: Root Mean Squared Error on class indices
        - ``qwk``: Quadratic Weighted Kappa
        - ``spearman``: Spearman rank correlation
    """
    return {
        "accuracy": ordinal_accuracy(predictions, targets, tolerance=0),
        "adjacent_accuracy": adjacent_accuracy(predictions, targets),
        "mae": ordinal_mae(predictions, targets),
        "rmse": ordinal_rmse(predictions, targets),
        "qwk": quadratic_weighted_kappa(predictions, targets, k),
        "spearman": spearman_correlation(predictions, targets),
    }


__all__ = [
    "mse_loss",
    "mae_loss",
    "rmse_loss",
    "cross_entropy_loss",
    "accuracy",
    # Ordinal metrics
    "ordinal_mae",
    "ordinal_rmse",
    "ordinal_accuracy",
    "adjacent_accuracy",
    "quadratic_weighted_kappa",
    "spearman_correlation",
    "ordinal_metrics",
]
