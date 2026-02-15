from __future__ import annotations

import math
from typing import NamedTuple

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


# ---------------------------------------------------------------------------
# Wristband Gaussian Loss
# ---------------------------------------------------------------------------


def _eps_for_dtype(dtype: torch.dtype, large: bool = False) -> float:
    """Return a small epsilon suitable for *dtype*.

    When *large* is True the returned value is ``sqrt(eps)`` -- useful as a
    variance floor where machine-epsilon itself would be too tight.
    """
    eps = torch.finfo(dtype).eps
    return math.sqrt(eps) if large else eps


def w2_to_standard_normal_sq(
    x: torch.Tensor, *, reduction: str = "mean"
) -> torch.Tensor:
    r"""Squared 2-Wasserstein distance between the Gaussian fit to *x* and N(0, I).

    .. math::

       W_2^2 = \|\mu\|^2 + \sum_i (\sqrt{\lambda_i} - 1)^2

    where :math:`\lambda_i` are eigenvalues of the sample covariance of *x*.

    Parameters
    ----------
    x : Tensor
        Shape ``(..., B, d)`` where ``B`` is the number of samples and ``d``
        the feature dimension.
    reduction : ``"none"`` | ``"mean"`` | ``"sum"``

    Returns
    -------
    Tensor
        ``(...)`` when ``reduction="none"``, scalar otherwise.
    """
    if x.ndim < 2:
        raise ValueError(
            f"Expected x.ndim>=2 with shape (..., B, d), got {tuple(x.shape)}"
        )
    b = x.shape[-2]
    d = x.shape[-1]
    if b < 2:
        raise ValueError("Need B>=2 for covariance (denominator B-1).")

    work_dtype = (
        torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    )
    xw = x.to(dtype=work_dtype)

    mu = xw.mean(dim=-2, keepdim=True)
    xc = xw - mu
    mu2 = mu.squeeze(-2).square().sum(dim=-1)
    denom = float(b - 1)

    if d <= b:
        m = (xc.transpose(-1, -2) @ xc) / denom
        m_dim = d
    else:
        m = (xc @ xc.transpose(-1, -2)) / denom
        m_dim = b

    m = 0.5 * (m + m.transpose(-1, -2))

    eig = torch.linalg.eigvalsh(m)
    eig = eig.clamp_min(0.0)

    sqrt_eig = torch.sqrt(eig + _eps_for_dtype(eig.dtype))
    bw2 = (sqrt_eig - 1.0).square().sum(dim=-1)

    if d > m_dim:
        bw2 = bw2 + (d - m_dim)

    loss = mu2 + bw2

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("reduction must be one of {'none','mean','sum'}")


class S_LossComponents(NamedTuple):
    """Named tuple returned by :class:`C_WristbandGaussianLoss`."""

    total: torch.Tensor
    rep: torch.Tensor
    rad: torch.Tensor
    ang: torch.Tensor
    mom: torch.Tensor


class C_WristbandGaussianLoss:
    r"""Batch loss encouraging :math:`x \sim \mathcal{N}(0, I)` via wristband
    repulsion on the (direction, radius) decomposition of samples, with optional
    marginal-uniformity and moment-matching penalties.

    The loss maps each sample to a *wristband* representation ``(u, t)`` where
    ``u`` is the unit direction and ``t = gammainc(d/2, ||x||^2/2)`` is the
    CDF-transformed radius (uniform under the null).  Repulsion is computed
    with reflecting boundary conditions in ``t`` (3-image method) and a
    configurable angular kernel on ``u``.

    All component losses are calibrated by Monte-Carlo sampling from the null
    distribution at construction time, so the returned ``total`` is a
    zero-mean, unit-variance z-score under :math:`\mathcal{N}(0, I)`.

    Parameters
    ----------
    beta : float
        Bandwidth parameter for the Gaussian kernel in the repulsion term.
    alpha : float | None
        Coupling constant between angular and radial scales.  ``None`` picks
        a heuristic default that balances the two.
    angular : ``"chordal"`` | ``"geodesic"``
        Metric on the unit sphere for the angular component.
    reduction : ``"per_point"`` | ``"global"``
        Whether repulsion is averaged per-row (per-point) or globally.
    lambda_rad, lambda_ang, lambda_mom : float
        Weights for the radial-uniformity, angular-uniformity, and moment
        penalty components.
    moment : str
        Moment penalty type. One of ``"mu_only"``, ``"kl_diag"``,
        ``"kl_full"``, ``"jeff_diag"``, ``"jeff_full"``, ``"w2"``.
    calibration_shape : tuple[int, int] | None
        ``(N, D)`` shape for Monte-Carlo calibration.  If provided the loss
        components are normalised to zero mean / unit variance under the null.
    calibration_reps : int
        Number of Monte-Carlo repetitions for calibration.
    calibration_device, calibration_dtype
        Device and dtype for calibration samples.

    References
    ----------
    Adapted from ``EmbedModels.py`` in https://github.com/mvparakhin/ml-tidbits

    Example
    -------
    >>> loss_fn = C_WristbandGaussianLoss(calibration_shape=(256, 8))
    >>> z = torch.randn(256, 8)
    >>> lc = loss_fn(z)
    >>> lc.total.backward()
    """

    def __init__(
        self,
        *,
        beta: float = 8.0,
        alpha: float | None = None,
        angular: str = "chordal",
        reduction: str = "per_point",
        lambda_rad: float = 0.1,
        lambda_ang: float = 0.0,
        moment: str = "w2",
        lambda_mom: float = 1.0,
        calibration_shape: tuple[int, int] | None = None,
        calibration_reps: int = 1024,
        calibration_device: str | torch.device = "cpu",
        calibration_dtype: torch.dtype = torch.float32,
    ):
        if beta <= 0:
            raise ValueError("beta must be > 0")
        if angular not in ("chordal", "geodesic"):
            raise ValueError("angular must be 'chordal' or 'geodesic'")
        if reduction not in ("per_point", "global"):
            raise ValueError("reduction must be 'per_point' or 'global'")
        if moment not in (
            "mu_only",
            "kl_diag",
            "kl_full",
            "jeff_diag",
            "jeff_full",
            "w2",
        ):
            raise ValueError(
                "moment must be 'mu_only', 'kl_diag', 'kl_full', "
                "'jeff_diag', 'jeff_full' or 'w2'"
            )

        self.beta = float(beta)
        self.angular = angular
        self.reduction = reduction

        if alpha is None:
            if angular == "chordal":
                alpha = math.sqrt(1.0 / 12.0)
            else:
                alpha = math.sqrt(2.0 / (3.0 * math.pi * math.pi))
        self.alpha = float(alpha)
        self.beta_alpha2 = self.beta * (self.alpha * self.alpha)

        self.lambda_rad = float(lambda_rad)
        self.lambda_ang = float(lambda_ang)
        self.moment = moment
        self.lambda_mom = float(lambda_mom)
        self.eps = 1.0e-12
        self.clamp_cos = 1.0e-6

        # Calibration statistics (identity transform when not calibrated)
        self.mean_rep = self.mean_rad = self.mean_ang = self.mean_mom = 0.0
        self.std_rep = self.std_rad = self.std_ang = self.std_mom = 1.0
        self.std_total = 1.0

        if calibration_shape is not None:
            self._calibrate(
                calibration_shape,
                calibration_reps,
                calibration_device,
                calibration_dtype,
            )

    # ---- calibration ----

    def _calibrate(
        self,
        shape: tuple[int, int],
        reps: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        n, d = shape
        if n < 2 or d < 1 or reps < 2:
            return

        all_rep: list[float] = []
        all_rad: list[float] = []
        all_ang: list[float] = []
        all_mom: list[float] = []

        sum_rep = sum_rad = sum_ang = sum_mom = 0.0
        sum2_rep = sum2_rad = sum2_ang = sum2_mom = 0.0

        with torch.no_grad():
            for _ in range(int(reps)):
                x_gauss = torch.randn(int(n), int(d), device=device, dtype=dtype)
                comp = self._compute(x_gauss)

                f_rep = float(comp.rep)
                f_rad = float(comp.rad)
                f_ang = float(comp.ang)
                f_mom = float(comp.mom)
                sum_rep += f_rep
                sum2_rep += f_rep * f_rep
                all_rep.append(f_rep)
                sum_rad += f_rad
                sum2_rad += f_rad * f_rad
                all_rad.append(f_rad)
                sum_ang += f_ang
                sum2_ang += f_ang * f_ang
                all_ang.append(f_ang)
                sum_mom += f_mom
                sum2_mom += f_mom * f_mom
                all_mom.append(f_mom)

        reps_f = float(reps)
        bessel = reps_f / (reps_f - 1.0)

        self.mean_rep = sum_rep / reps_f
        self.mean_rad = sum_rad / reps_f
        self.mean_ang = sum_ang / reps_f
        self.mean_mom = sum_mom / reps_f

        var_rep = (sum2_rep / reps_f - self.mean_rep * self.mean_rep) * bessel
        var_rad = (sum2_rad / reps_f - self.mean_rad * self.mean_rad) * bessel
        var_ang = (sum2_ang / reps_f - self.mean_ang * self.mean_ang) * bessel
        var_mom = (sum2_mom / reps_f - self.mean_mom * self.mean_mom) * bessel

        eps_cal = float(_eps_for_dtype(dtype, True))
        self.std_rep = math.sqrt(max(var_rep, eps_cal))
        self.std_rad = math.sqrt(max(var_rad, eps_cal))
        self.std_ang = math.sqrt(max(var_ang, eps_cal))
        self.std_mom = math.sqrt(max(var_mom, eps_cal))

        # Std of the weighted total (for final normalisation)
        sum_total = sum2_total = 0.0
        for i in range(int(reps)):
            t_rep = (all_rep[i] - self.mean_rep) / self.std_rep
            t_rad = self.lambda_rad * (all_rad[i] - self.mean_rad) / self.std_rad
            t_ang = self.lambda_ang * (all_ang[i] - self.mean_ang) / self.std_ang
            t_mom = self.lambda_mom * (all_mom[i] - self.mean_mom) / self.std_mom
            total = t_rep + t_rad + t_ang + t_mom
            sum_total += total
            sum2_total += total * total

        mean_total = sum_total / reps_f
        var_total = (sum2_total / reps_f - mean_total * mean_total) * bessel
        self.std_total = math.sqrt(max(var_total, eps_cal))

    # ---- core computation ----

    def _compute(self, x: torch.Tensor) -> S_LossComponents:
        if x.ndim < 2:
            raise ValueError(
                f"Expected x.ndim>=2 with shape (..., N, D), got {tuple(x.shape)}"
            )

        n = int(x.shape[-2])
        d = int(x.shape[-1])
        batch_shape = x.shape[:-2]

        if n < 2 or d < 1:
            z = x.sum(dim=(-2, -1)) * 0.0
            return S_LossComponents(z, z, z, z, z)

        wdtype = (
            torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        )
        xw = x.to(wdtype)
        n_f, d_f = float(n), float(d)
        beta, eps = self.beta, self.eps

        mu = xw.mean(dim=-2)
        xc = xw - mu[..., None, :]

        # ---- moment penalty ----
        mom_pen = xw.new_zeros(batch_shape)
        if self.lambda_mom != 0.0:
            if self.moment == "w2":
                mom_pen = w2_to_standard_normal_sq(xw, reduction="none") / d_f
            elif self.moment == "jeff_diag":
                var = xc.square().sum(dim=-2) / (n_f - 1.0)
                v = var + eps
                inv_v = v.reciprocal()
                mu2 = mu.square()
                mom_pen = 0.25 * (v + inv_v + mu2 + mu2 * inv_v - 2.0).mean(dim=-1)
            elif self.moment == "jeff_full":
                eps_cov = (
                    max(eps, 1.0e-6)
                    if wdtype == torch.float32
                    else max(eps, float(torch.finfo(wdtype).eps))
                )
                cov = (xc.transpose(-1, -2) @ xc) / (n_f - 1.0)
                eye = torch.eye(d, device=xw.device, dtype=wdtype)
                cov = cov + eps_cov * eye
                chol, info = torch.linalg.cholesky_ex(cov)
                tr = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                inv_cov = torch.cholesky_solve(eye, chol)
                tr_inv = inv_cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                mu_col = mu[..., :, None]
                sol_mu = torch.cholesky_solve(mu_col, chol)
                mu_inv_mu = (mu_col * sol_mu).sum(dim=(-2, -1))
                mu2_sum = mu.square().sum(dim=-1)
                mom_pen = (
                    0.25 * (tr + tr_inv + mu2_sum + mu_inv_mu - 2.0 * d_f) / d_f
                )
            elif self.moment == "mu_only":
                mom_pen = mu.square().mean(dim=-1)
            elif self.moment == "kl_diag":
                var = xc.square().sum(dim=-2) / (n_f - 1.0)
                mom_pen = 0.5 * (var + mu.square() - 1.0 - torch.log(var + eps)).mean(
                    dim=-1
                )
            else:  # kl_full
                eye = torch.eye(d, device=xw.device, dtype=wdtype)
                cov = (xc.transpose(-1, -2) @ xc) / n_f + eps * eye
                chol, info = torch.linalg.cholesky_ex(cov)
                diag = chol.diagonal(dim1=-2, dim2=-1)
                logdet = 2.0 * torch.log(diag).sum(dim=-1)
                tr = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                mu2 = mu.square().sum(dim=-1)
                mom_pen = 0.5 * (tr + mu2 - d_f - logdet) / d_f

        # ---- wristband map (u, t) ----
        s = xw.square().sum(dim=-1).clamp_min(eps)
        u = xw * torch.rsqrt(s)[..., :, None]
        a_df = s.new_tensor(0.5 * d_f)
        t = torch.special.gammainc(a_df, 0.5 * s).clamp(eps, 1.0 - eps)

        # ---- radial 1D W2^2 on t vs Unif(0,1) ----
        rad_loss = xw.new_zeros(batch_shape)
        if self.lambda_rad != 0.0:
            t_sorted, _ = torch.sort(t, dim=-1)
            q = (torch.arange(n, device=xw.device, dtype=wdtype) + 0.5) / n_f
            rad_loss = 12.0 * (t_sorted - q).square().mean(dim=-1)

        # ---- angular kernel exponent ----
        g = (u @ u.transpose(-1, -2)).clamp(-1.0, 1.0)

        if self.angular == "chordal":
            e_ang = (2.0 * self.beta_alpha2) * (g - 1.0)
        else:
            theta = torch.acos(g.clamp(-1.0 + self.clamp_cos, 1.0 - self.clamp_cos))
            ang2 = theta.square()
            ang2 = ang2 - torch.diag_embed(
                ang2.diagonal(dim1=-2, dim2=-1)
            )
            e_ang = -self.beta_alpha2 * ang2

        # ---- optional angular-only uniformity ----
        ang_loss = xw.new_zeros(batch_shape)
        if self.lambda_ang != 0.0:
            if self.reduction == "per_point":
                row_sum = torch.exp(e_ang).sum(dim=-1) - 1.0
                mean_k = row_sum / (n_f - 1.0)
                ang_loss = torch.log(mean_k + eps).mean(dim=-1) / beta
            else:
                total = torch.exp(e_ang).sum(dim=(-2, -1)) - n_f
                mean_k = total / (n_f * (n_f - 1.0))
                ang_loss = torch.log(mean_k + eps) / beta

        # ---- 3-image reflected kernel for joint (u, t) repulsion ----
        tc = t[..., :, None]
        tr_ = t[..., None, :]
        diff0 = tc - tr_
        diff1 = tc + tr_
        diff2 = diff1 - 2.0

        if self.reduction == "per_point":
            row_sum = torch.exp(
                torch.addcmul(e_ang, diff0, diff0, value=-beta)
            ).sum(dim=-1)
            row_sum += torch.exp(
                torch.addcmul(e_ang, diff1, diff1, value=-beta)
            ).sum(dim=-1)
            row_sum += torch.exp(
                torch.addcmul(e_ang, diff2, diff2, value=-beta)
            ).sum(dim=-1)
            row_sum -= 1.0
            mean_k = row_sum / (3.0 * n_f - 1.0)
            rep_loss = torch.log(mean_k + eps).mean(dim=-1) / beta
        else:
            total = torch.exp(
                torch.addcmul(e_ang, diff0, diff0, value=-beta)
            ).sum(dim=(-2, -1))
            total += torch.exp(
                torch.addcmul(e_ang, diff1, diff1, value=-beta)
            ).sum(dim=(-2, -1))
            total += torch.exp(
                torch.addcmul(e_ang, diff2, diff2, value=-beta)
            ).sum(dim=(-2, -1))
            total -= n_f
            mean_k = total / (3.0 * n_f * n_f - n_f)
            rep_loss = torch.log(mean_k + eps) / beta

        return S_LossComponents(rep_loss, rep_loss, rad_loss, ang_loss, mom_pen)

    # ---- public interface ----

    def __call__(self, x: torch.Tensor) -> S_LossComponents:
        """Compute the calibrated wristband-Gaussian loss.

        Parameters
        ----------
        x : Tensor of shape ``(..., N, D)``
            Batch of samples (``N`` samples of dimension ``D``).

        Returns
        -------
        S_LossComponents
            Named tuple ``(total, rep, rad, ang, mom)`` where ``total`` is the
            scalar to back-propagate and the rest are normalised diagnostics.
        """
        comp = self._compute(x)

        norm_rep = (comp.rep - self.mean_rep) / self.std_rep
        norm_rad = (comp.rad - self.mean_rad) / self.std_rad
        norm_ang = (comp.ang - self.mean_ang) / self.std_ang
        norm_mom = (comp.mom - self.mean_mom) / self.std_mom

        total = (
            norm_rep
            + self.lambda_rad * norm_rad
            + self.lambda_ang * norm_ang
            + self.lambda_mom * norm_mom
        ) / self.std_total

        return S_LossComponents(
            total.mean(), norm_rep.mean(), norm_rad.mean(), norm_ang.mean(), norm_mom.mean()
        )


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
    # Wristband Gaussian loss
    "w2_to_standard_normal_sq",
    "S_LossComponents",
    "C_WristbandGaussianLoss",
]
