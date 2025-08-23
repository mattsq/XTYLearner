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

__all__ = ["LowRankDiagHead"]
