import math
import torch
import torch.nn.functional as func


def nll_lowrank_diag(
    y: torch.Tensor,
    mu: torch.Tensor,
    F: torch.Tensor,
    sigma2: torch.Tensor,
    jitter: float = 1e-6,
    max_tries: int = 5,
) -> torch.Tensor:
    """Negative log-likelihood of a Gaussian with low-rank plus diagonal covariance.

    The covariance is ``F F^T + diag(sigma2)``.  Small ``jitter`` is added to the
    low-rank term to guarantee positive-definiteness; if the Cholesky
    decomposition still fails we exponentially increase the jitter.
    """

    B, d_y = y.shape
    r = F.size(-1)
    diff = (y - mu).unsqueeze(-1)
    F = torch.nan_to_num(F)
    sigma2 = torch.nan_to_num(sigma2).clamp_min(jitter)
    D_inv = 1.0 / sigma2
    Ft_Dinv = torch.nan_to_num(F.transpose(1, 2) * D_inv.unsqueeze(-2))
    eye = torch.eye(r, device=y.device, dtype=y.dtype).expand(B, r, r)
    M = torch.nan_to_num(torch.baddbmm(eye, Ft_Dinv, F))

    jitter_i = jitter
    L = None
    logdet_M = None
    for _ in range(max_tries):
        M_j = torch.nan_to_num(M + jitter_i * eye)
        L, info = torch.linalg.cholesky_ex(M_j)
        if (info == 0).all():
            logdet_M = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
            break
        jitter_i *= 10.0

    if L is not None and logdet_M is not None:
        rhs = Ft_Dinv @ diff
        sol = torch.cholesky_solve(rhs, L)
    else:
        # As a last resort fall back to a pseudoinverse.
        M_j = torch.nan_to_num(M + jitter_i * eye)
        M_inv = torch.linalg.pinv(M_j)
        rhs = Ft_Dinv @ diff
        sol = M_inv @ rhs
        sign, logabsdet = torch.linalg.slogdet(M_j)
        logdet_M = logabsdet

    alpha = D_inv.unsqueeze(-1) * diff - (D_inv.unsqueeze(-1) * F @ sol)
    quad = (diff.squeeze(-1) * alpha.squeeze(-1)).sum(-1)
    logdet = sigma2.log().sum(-1) + logdet_M
    return 0.5 * (logdet + quad + d_y * math.log(2 * math.pi))


# ---------------------------------------------------------------------------
# Ordinal Classification Losses
# ---------------------------------------------------------------------------


def cumulative_link_loss(
    cumprobs: torch.Tensor,
    target: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Negative log-likelihood for cumulative link (proportional odds) model.

    Parameters
    ----------
    cumprobs
        Cumulative probabilities P(T <= j) of shape ``(batch, k-1)``.
    target
        Integer class labels in ``{0, 1, ..., k-1}`` of shape ``(batch,)``.
    k
        Number of ordinal classes.

    Returns
    -------
    torch.Tensor
        Scalar mean negative log-likelihood.
    """
    batch_size = target.size(0)
    target = target.long()

    # Compute class probabilities from cumulative probabilities
    # P(T=j) = P(T<=j) - P(T<=j-1), with P(T<=-1)=0 and P(T<=k-1)=1
    zeros = torch.zeros(batch_size, 1, device=cumprobs.device, dtype=cumprobs.dtype)
    ones = torch.ones(batch_size, 1, device=cumprobs.device, dtype=cumprobs.dtype)
    cumprobs_padded = torch.cat([zeros, cumprobs, ones], dim=1)  # (batch, k+1)

    # P(T=j) = P(T<=j) - P(T<=j-1)
    class_probs = cumprobs_padded[:, 1:] - cumprobs_padded[:, :-1]  # (batch, k)
    class_probs = class_probs.clamp(min=1e-7)  # Numerical stability

    # Gather the probability of the true class
    target_probs = class_probs.gather(1, target.unsqueeze(1)).squeeze(1)

    return -torch.log(target_probs).mean()


def coral_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """CORAL (Consistent Rank Logits) loss for ordinal regression.

    Uses k-1 binary classifiers with shared weights to predict P(T > j).

    Parameters
    ----------
    logits
        Raw logits of shape ``(batch, k-1)`` where ``logits[:, j]``
        represents the logit for P(T > j).
    target
        Integer class labels in ``{0, 1, ..., k-1}`` of shape ``(batch,)``.
    k
        Number of ordinal classes.

    Returns
    -------
    torch.Tensor
        Scalar mean binary cross-entropy loss.

    References
    ----------
    Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal
    regression for neural networks with application to age estimation.
    Pattern Recognition Letters.
    """
    batch_size = target.size(0)
    target = target.long()

    # Create binary labels: for class c, labels are [1,1,...,1,0,0,...,0]
    # where the first c positions are 1 (T > 0, T > 1, ..., T > c-1)
    levels = torch.arange(k - 1, device=target.device).unsqueeze(0)  # (1, k-1)
    binary_targets = (target.unsqueeze(1) > levels).float()  # (batch, k-1)

    # Binary cross-entropy for each threshold
    loss = func.binary_cross_entropy_with_logits(logits, binary_targets, reduction="mean")
    return loss


def ordinal_regression_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Soft-label ordinal loss with Gaussian-weighted targets.

    Creates soft labels where adjacent classes receive partial weight,
    encouraging the model to respect ordinal structure.

    Parameters
    ----------
    logits
        Raw class logits of shape ``(batch, k)``.
    target
        Integer class labels of shape ``(batch,)``.
    alpha
        Controls the spread of soft labels. Higher values create
        sharper distributions (more like hard labels).

    Returns
    -------
    torch.Tensor
        Scalar soft cross-entropy loss.
    """
    k = logits.size(-1)
    target = target.long()

    # Create soft labels with Gaussian weighting
    class_indices = torch.arange(k, device=target.device, dtype=logits.dtype)
    distances = (class_indices.unsqueeze(0) - target.unsqueeze(1).float()) ** 2
    soft_labels = torch.exp(-alpha * distances)
    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)  # Normalize

    # Soft cross-entropy
    log_probs = func.log_softmax(logits, dim=-1)
    loss = -(soft_labels * log_probs).sum(dim=-1).mean()
    return loss


def ordinal_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Focal loss variant for ordinal classification.

    Combines focal loss (down-weighting easy examples) with ordinal
    soft labels to handle class imbalance in ordinal problems.

    Parameters
    ----------
    logits
        Raw class logits of shape ``(batch, k)``.
    target
        Integer class labels of shape ``(batch,)``.
    gamma
        Focal loss focusing parameter. Higher values focus more on
        hard examples.
    alpha
        Controls the spread of ordinal soft labels.

    Returns
    -------
    torch.Tensor
        Scalar focal ordinal loss.
    """
    k = logits.size(-1)
    target = target.long()

    # Create soft labels
    class_indices = torch.arange(k, device=target.device, dtype=logits.dtype)
    distances = (class_indices.unsqueeze(0) - target.unsqueeze(1).float()) ** 2
    soft_labels = torch.exp(-alpha * distances)
    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)

    # Focal weighting
    probs = func.softmax(logits, dim=-1)
    log_probs = func.log_softmax(logits, dim=-1)

    # Focal weight: (1 - p)^gamma where p is probability of true class
    focal_weight = (1 - probs) ** gamma

    loss = -(focal_weight * soft_labels * log_probs).sum(dim=-1).mean()
    return loss


# ---------------------------------------------------------------------------
# Ordinal Probability Utilities
# ---------------------------------------------------------------------------


def cumulative_to_class_probs(cumprobs: torch.Tensor) -> torch.Tensor:
    """Convert cumulative probabilities P(T<=j) to class probabilities P(T=j).

    Parameters
    ----------
    cumprobs
        Cumulative probabilities of shape ``(batch, k-1)``.

    Returns
    -------
    torch.Tensor
        Class probabilities of shape ``(batch, k)``.
    """
    batch_size = cumprobs.size(0)
    zeros = torch.zeros(batch_size, 1, device=cumprobs.device, dtype=cumprobs.dtype)
    ones = torch.ones(batch_size, 1, device=cumprobs.device, dtype=cumprobs.dtype)
    cumprobs_padded = torch.cat([zeros, cumprobs, ones], dim=1)
    class_probs = cumprobs_padded[:, 1:] - cumprobs_padded[:, :-1]
    return class_probs.clamp(min=0)


def class_probs_to_cumulative(probs: torch.Tensor) -> torch.Tensor:
    """Convert class probabilities P(T=j) to cumulative probabilities P(T<=j).

    Parameters
    ----------
    probs
        Class probabilities of shape ``(batch, k)``.

    Returns
    -------
    torch.Tensor
        Cumulative probabilities of shape ``(batch, k-1)``.
    """
    cumprobs = torch.cumsum(probs, dim=-1)
    return cumprobs[:, :-1]  # Exclude P(T<=k-1) = 1


def coral_logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORAL logits to class probabilities.

    Parameters
    ----------
    logits
        CORAL logits of shape ``(batch, k-1)`` representing logit(P(T > j)).

    Returns
    -------
    torch.Tensor
        Class probabilities of shape ``(batch, k)``.
    """
    # P(T > j) = sigmoid(logits)
    exceed_probs = torch.sigmoid(logits)  # (batch, k-1)

    # P(T=0) = 1 - P(T>0)
    # P(T=j) = P(T>j-1) - P(T>j) for j > 0
    # P(T=k-1) = P(T>k-2)
    batch_size = logits.size(0)
    ones = torch.ones(batch_size, 1, device=logits.device, dtype=logits.dtype)
    zeros = torch.zeros(batch_size, 1, device=logits.device, dtype=logits.dtype)

    exceed_padded = torch.cat([ones, exceed_probs, zeros], dim=1)  # (batch, k+1)
    class_probs = exceed_padded[:, :-1] - exceed_padded[:, 1:]  # (batch, k)
    return class_probs.clamp(min=0)


def ordinal_predict(
    output: torch.Tensor,
    method: str = "mode",
    output_type: str = "logits",
) -> torch.Tensor:
    """Predict ordinal class from model output.

    Parameters
    ----------
    output
        Model output. Interpretation depends on ``output_type``.
    method
        Prediction method:
        - ``"mode"``: Most likely class (argmax).
        - ``"mean"``: Expected value E[T], rounded to nearest int.
        - ``"median"``: Class where cumulative probability crosses 0.5.
    output_type
        Type of output:
        - ``"logits"``: Standard class logits (batch, k).
        - ``"cumulative"``: Cumulative probabilities P(T<=j) (batch, k-1).
        - ``"coral"``: CORAL logits for P(T>j) (batch, k-1).

    Returns
    -------
    torch.Tensor
        Predicted class labels of shape ``(batch,)``.
    """
    # Convert to class probabilities
    if output_type == "logits":
        probs = func.softmax(output, dim=-1)
    elif output_type == "cumulative":
        probs = cumulative_to_class_probs(output)
    elif output_type == "coral":
        probs = coral_logits_to_class_probs(output)
    else:
        raise ValueError(f"Unknown output_type: {output_type}")

    k = probs.size(-1)

    if method == "mode":
        return probs.argmax(dim=-1)
    elif method == "mean":
        class_indices = torch.arange(k, device=probs.device, dtype=probs.dtype)
        expected = (probs * class_indices).sum(dim=-1)
        return expected.round().long().clamp(0, k - 1)
    elif method == "median":
        cumprobs = torch.cumsum(probs, dim=-1)
        # Find first class where cumulative prob >= 0.5
        return (cumprobs >= 0.5).long().argmax(dim=-1)
    else:
        raise ValueError(f"Unknown method: {method}")


__all__ = [
    "nll_lowrank_diag",
    # Ordinal losses
    "cumulative_link_loss",
    "coral_loss",
    "ordinal_regression_loss",
    "ordinal_focal_loss",
    # Ordinal utilities
    "cumulative_to_class_probs",
    "class_probs_to_cumulative",
    "coral_logits_to_class_probs",
    "ordinal_predict",
]
