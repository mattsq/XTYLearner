import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model


def notears_acyclicity(A: torch.Tensor) -> torch.Tensor:
    """Smooth acyclicity constraint from Zheng et al."""
    return torch.trace(torch.matrix_exp(A * A)) - A.size(0)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = make_mlp([in_dim, hidden, hidden, out_dim])

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(z)


@register_model("gnn_scm")
class GNN_SCM(nn.Module):
    """Graph Neural Structural Causal Model."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int | None = None,
        noise_dim: int = 4,
        *,
        hidden: int = 128,
        forbid_y_to_x: bool = True,
        lambda_acyc: float = 10.0,
        gamma_l1: float = 1e-2,
    ) -> None:
        super().__init__()
        self.k = k
        self.d_y = d_y
        self.noise_dim = noise_dim
        self.d_nodes = d_x + 2  # X nodes + T + Y

        B = torch.zeros(self.d_nodes, self.d_nodes)
        mask = torch.ones_like(B)
        if forbid_y_to_x:
            mask[-1, :d_x] = 0
        mask.fill_diagonal_(0)
        self.register_buffer("mask", mask)
        self.B = nn.Parameter(B)

        out_dim_T = k if k is not None else 2
        self.f_T = _MLP(d_x + noise_dim, out_dim_T, hidden)
        self.f_Y = _MLP(d_x + (k or 1) + noise_dim, d_y * 2, hidden)

        self.lambda_acyc = lambda_acyc
        self.gamma_l1 = gamma_l1

    # --------------------------------------------------------------
    def _A(self) -> torch.Tensor:
        return torch.sigmoid(self.B) * self.mask

    def sample_T(self, x: torch.Tensor) -> torch.Tensor:
        """Draw a treatment sample ``t`` from ``p(t|x)``."""
        eps = torch.randn(x.size(0), self.noise_dim, device=x.device)
        pars = self.f_T(torch.cat([x, eps], -1))
        if self.k is not None:
            dist = torch.distributions.Categorical(logits=pars)
            return dist.sample()
        mu, log_sigma = pars.chunk(2, -1)
        return mu + torch.exp(log_sigma) * torch.randn_like(mu)

    def sample_Y(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample an outcome ``y`` conditional on ``x`` and treatment ``t``."""
        eps = torch.randn(x.size(0), self.noise_dim, device=x.device)
        t_in = F.one_hot(t, self.k).float() if self.k is not None else t.unsqueeze(-1)
        mu, log_sigma = self.f_Y(torch.cat([x, t_in, eps], -1)).chunk(2, -1)
        return mu + torch.exp(log_sigma) * torch.randn_like(mu)

    # --------------------------------------------------------------
    def log_p_t(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Log-probability of treatment assignments under the model."""
        eps = torch.zeros(x.size(0), self.noise_dim, device=x.device)
        pars = self.f_T(torch.cat([x, eps], -1))
        if self.k is not None:
            return -F.cross_entropy(pars, t, reduction="none")
        mu, log_sigma = pars.chunk(2, -1)
        return -0.5 * ((t - mu) / log_sigma.exp()).pow(2) - log_sigma

    def log_p_y(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability of outcomes given ``x`` and ``t``."""
        t_in = F.one_hot(t, self.k).float() if self.k is not None else t.unsqueeze(-1)
        eps = torch.zeros(x.size(0), self.noise_dim, device=x.device)
        mu, log_sigma = self.f_Y(torch.cat([x, t_in, eps], -1)).chunk(2, -1)
        return -0.5 * ((y - mu) / log_sigma.exp()).pow(2) - log_sigma

    # --------------------------------------------------------------
    def loss(self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
        A = self._A()
        acyc = notears_acyclicity(A)
        labelled = t_obs >= 0
        loglike = torch.tensor(0.0, device=x.device)
        if labelled.any():
            loglike += self.log_p_t(x[labelled], t_obs[labelled]).mean()
            loglike += self.log_p_y(x[labelled], t_obs[labelled], y[labelled]).mean()
        unlabel = ~labelled
        if unlabel.any():
            t = self.sample_T(x[unlabel]).detach()
            loglike += self.log_p_y(x[unlabel], t, y[unlabel]).mean()
        l1 = (A.abs() * self.mask).sum()
        return -loglike + self.lambda_acyc * acyc + self.gamma_l1 * l1

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        """Return the mean outcome for covariates ``x`` under treatment ``t``."""
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long if self.k is not None else torch.float32, device=x.device)
        t_in = F.one_hot(t, self.k).float() if self.k is not None else t.unsqueeze(-1)
        eps = torch.zeros(x.size(0), self.noise_dim, device=x.device)
        mu, _ = self.f_Y(torch.cat([x, t_in, eps], -1)).chunk(2, -1)
        return mu.squeeze(-1)

    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Compute ``p(t|x)`` (or parameters of the treatment distribution)."""
        eps = torch.zeros(x.size(0), self.noise_dim, device=x.device)
        pars = self.f_T(torch.cat([x, eps], -1))
        if self.k is None:
            mu, log_sigma = pars.chunk(2, -1)
            return torch.cat([mu, log_sigma.exp()], dim=-1)
        return F.softmax(pars, dim=-1)


__all__ = ["GNN_SCM"]
