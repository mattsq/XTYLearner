"""Energy-Based GNN-SCM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .layers import make_mlp
from .gnn_scm import notears_acyclicity


class MaskedGraphConv(nn.Module):
    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:  # noqa: D401
        msg = torch.einsum("ij,bjd->bid", A, h)
        return h + msg / h.size(1)


class MaskedGraphConvStack(nn.Module):
    def __init__(self, d_nodes: int, d_in: int, hidden: int = 128) -> None:
        super().__init__()
        self.d_nodes = d_nodes
        self.fc_in = nn.Linear(d_in, hidden)
        self.conv1 = MaskedGraphConv()
        self.conv2 = MaskedGraphConv()

    def forward(self, z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(z)
        h = h.unsqueeze(1).repeat(1, self.d_nodes, 1)
        h = F.relu(self.conv1(h, A))
        h = F.relu(self.conv2(h, A))
        return h


@register_model("gnn_ebm")
class GNN_EBM(nn.Module):
    """Energy-based GNN-SCM with a single scalar energy."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k_t: int | None = None,
        *,
        k: int | None = None,
        hidden: int = 128,
        k_langevin: int = 20,
        eta: float = 1e-2,
        lambda_acyc: float = 5.0,
        gamma_l1: float = 1e-2,
    ) -> None:
        if k_t is None:
            k_t = k
        elif k is not None:
            k_t = k
        super().__init__()
        self.d_x = d_x
        self.k_t = k_t
        self.d_y = d_y
        self.d_nodes = d_x + 2

        mask = torch.ones(self.d_nodes, self.d_nodes)
        mask[-1, :d_x] = 0
        mask.fill_diagonal_(0)
        self.register_buffer("mask", mask)
        self.B = nn.Parameter(torch.zeros_like(mask))

        d_in = d_x + 1 + d_y  # x + t + y
        self.gnn = MaskedGraphConvStack(self.d_nodes, d_in, hidden)
        self.e_T = make_mlp([hidden, hidden, 1])
        self.e_Y = make_mlp([hidden, hidden, 1])

        self.k_langevin = k_langevin
        self.eta = eta
        self.lambda_acyc = lambda_acyc
        self.gamma_l1 = gamma_l1

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.float32, device=x.device)
        else:
            t = t.view(-1).float()
        y_init = torch.zeros(x.size(0), self.d_y, device=x.device)
        _, y = self.langevin(x, t, y_init)
        return y

    @torch.no_grad()
    def predict(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        return self.predict_outcome(x, t)

    # ------------------------------------------------------------------
    def _A(self) -> torch.Tensor:
        return torch.sigmoid(self.B) * self.mask

    # ------------------------------------------------------------------
    def energy(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        A = self._A()
        h = self.gnn(torch.cat([x, t, y], dim=-1), A)
        e = self.e_T(h[:, -2]) + self.e_Y(h[:, -1])
        return e.squeeze(-1)

    # ------------------------------------------------------------------
    def langevin(
        self, x: torch.Tensor, t_init: torch.Tensor | None, y_init: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if t_init is not None:
            t0 = t_init.unsqueeze(-1).float()
        else:
            t0 = torch.randn(x.size(0), 1, device=x.device)
        z = torch.cat([t0, y_init], dim=-1).detach()
        z.requires_grad_(True)
        for _ in range(self.k_langevin):
            with torch.enable_grad():
                E = self.energy(x, z[:, :1], z[:, 1:])
                (grad,) = torch.autograd.grad(E.sum(), z)
            z = z - 0.5 * self.eta * grad
            z = z + torch.sqrt(
                torch.tensor(self.eta, device=x.device)
            ) * torch.randn_like(z)
            z = z.detach().requires_grad_(True)
        return z[:, :1].detach(), z[:, 1:].detach()

    # ------------------------------------------------------------------
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor | None
    ) -> torch.Tensor:
        pos_E = torch.tensor(0.0, device=x.device)
        if t_obs is not None:
            pos_E = self.energy(x, t_obs.unsqueeze(-1).float(), y).mean()
        t_neg, y_neg = self.langevin(x, t_obs, y)
        neg_E = self.energy(x, t_neg, y_neg).mean()
        A = self._A()
        return (
            pos_E
            - neg_E
            + self.lambda_acyc * notears_acyclicity(A)
            + self.gamma_l1 * (A.abs() * self.mask).sum()
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_missing_t(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t, _ = self.langevin(x, None, y)
        return t.squeeze(-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:  # noqa: F811
        """Approximate ``p(t|x,y)`` using the energy function."""
        if self.k_t is None:
            raise ValueError("Number of treatment classes 'k_t' must be set")

        logits = []
        num_classes = self.k_t if self.k_t > 1 else 2
        for i in range(num_classes):
            t_val = torch.full((x.size(0), 1), float(i), device=x.device)
            logits.append(-self.energy(x, t_val, y))
        logits = torch.stack(logits, dim=-1)
        return logits.softmax(dim=-1)


__all__ = ["GNN_EBM"]
