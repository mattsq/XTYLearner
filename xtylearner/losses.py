import math
import torch


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
    sigma2 = sigma2.clamp_min(jitter)
    D_inv = 1.0 / sigma2
    Ft_Dinv = F.transpose(1, 2) * D_inv.unsqueeze(-2)
    eye = torch.eye(r, device=y.device, dtype=y.dtype).expand(B, r, r)
    M = torch.baddbmm(eye, Ft_Dinv, F)

    jitter_i = jitter
    L = None
    logdet_M = None
    for _ in range(max_tries):
        M_j = M + jitter_i * eye
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
        M_j = M + jitter_i * eye
        M_inv = torch.linalg.pinv(M_j)
        rhs = Ft_Dinv @ diff
        sol = M_inv @ rhs
        sign, logabsdet = torch.linalg.slogdet(M_j)
        logdet_M = logabsdet

    alpha = D_inv.unsqueeze(-1) * diff - (D_inv.unsqueeze(-1) * F @ sol)
    quad = (diff.squeeze(-1) * alpha.squeeze(-1)).sum(-1)
    logdet = sigma2.log().sum(-1) + logdet_M
    return 0.5 * (logdet + quad + d_y * math.log(2 * math.pi))


__all__ = ["nll_lowrank_diag"]
