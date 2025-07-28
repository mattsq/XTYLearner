import math
import torch


def nll_lowrank_diag(y: torch.Tensor, mu: torch.Tensor, F: torch.Tensor, sigma2: torch.Tensor):
    """Negative log-likelihood of a Gaussian with low-rank plus diagonal covariance."""
    B, d_y = y.shape
    r = F.size(-1)
    diff = (y - mu).unsqueeze(-1)
    D_inv = 1.0 / sigma2
    Ft_Dinv = F.transpose(1, 2) * D_inv.unsqueeze(-2)
    M = torch.baddbmm(torch.eye(r, device=y.device).expand(B, r, r), Ft_Dinv, F)
    M_inv = torch.linalg.inv(M)
    alpha = D_inv.unsqueeze(-1) * diff - (D_inv.unsqueeze(-1) * F @ (M_inv @ (Ft_Dinv @ diff)))
    quad = (diff.squeeze(-1) * alpha.squeeze(-1)).sum(-1)
    logdet = sigma2.log().sum(-1) + torch.logdet(M)
    return 0.5 * (logdet + quad + d_y * math.log(2 * math.pi))

__all__ = ["nll_lowrank_diag"]
