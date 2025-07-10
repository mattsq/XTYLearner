"""Energy-Based Joint Model over (X, T, Y)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model


@register_model("joint_ebm")
class JointEBM(nn.Module):
    """Energy-based model of the joint distribution ``(X, T, Y)``.

    Training uses a contrastive objective on labelled rows and a
    marginalised objective on unlabelled rows.
    """

    def __init__(self, d_x: int, d_y: int, k: int = 2, hidden: int = 128) -> None:
        super().__init__()
        self.k = k
        self.d_y = d_y
        self.energy_net = make_mlp([d_x + d_y, hidden, hidden, k])

    def energy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, y], dim=-1)
        return self.energy_net(inp)

    # --------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, steps: int = 20, lr: float = 0.1
    ) -> torch.Tensor:
        """Approximate ``p(y|x,t)`` via gradient-based energy minimisation."""

        with torch.enable_grad():
            y = torch.zeros(x.size(0), self.d_y, device=x.device, requires_grad=True)
            for _ in range(steps):
                e_all = self.energy_net(torch.cat([x, y], dim=-1))
                e = e_all.gather(1, t.view(-1, 1).clamp_min(0))
                grad = torch.autograd.grad(e.sum(), y, create_graph=False)[0]
                y = (y - lr * grad).detach().requires_grad_(True)
        return y.detach()

    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        energies = self.energy_net(torch.cat([x, y], dim=-1))  # (B,2)
        labelled = t_obs >= 0
        loss_lab = torch.tensor(0.0, device=x.device)
        if labelled.any():
            t_lab = t_obs[labelled]
            loss_lab = F.cross_entropy(-energies[labelled], t_lab)
        unlabelled = ~labelled
        loss_ulb = torch.tensor(0.0, device=x.device)
        if unlabelled.any():
            lse = torch.logsumexp(-energies[unlabelled], dim=-1)
            loss_ulb = -lse.mean()
        return loss_lab + loss_ulb

    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        energies = self.energy_net(torch.cat([x, y], dim=-1))
        return F.softmax(-energies, dim=-1)


__all__ = ["JointEBM"]
