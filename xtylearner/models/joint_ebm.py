"""Energy-Based Joint Model over (X, T, Y).

This implementation models a discrete treatment ``T`` with ``k`` possible
levels. The energy network is a simple multilayer perceptron optionally wrapped
with spectral normalisation to stabilise training. A learnable temperature
parameter controls the sharpness of the energy distribution used when
predicting treatment probabilities or computing the training loss.
"""

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

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int = 2,
        hidden: int = 128,
        *,
        spectral_norm: bool = False,
        temperature: float = 1.0,
        learn_tau: bool = False,
        beta_reg: float = 1e-4,
    ) -> None:
        """Create a JointEBM instance.

        Parameters
        ----------
        d_x, d_y : int
            Dimensions of ``X`` and ``Y``.
        k : int, optional
            Number of treatment levels.
        hidden : int, optional
            Width of hidden layers in the energy network.
        spectral_norm : bool, optional
            If ``True``, apply spectral normalisation to all linear layers.
        temperature : float, optional
            Initial softmax temperature used for the treatment distribution.
        learn_tau : bool, optional
            If ``True``, ``temperature`` becomes a learnable parameter.
        """

        super().__init__()
        self.k = k
        self.d_y = d_y
        self.beta_reg = beta_reg
        net = make_mlp([d_x + d_y, hidden, hidden, k])
        if spectral_norm:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.utils.parametrizations.spectral_norm(m)
        self.energy_net = net

        tau = torch.tensor(float(temperature))
        if learn_tau:
            self.log_tau = nn.Parameter(tau.log())
        else:
            self.register_buffer("log_tau", tau.log())

        self.aux_mu = make_mlp([d_x + k, hidden, hidden, d_y])

    @torch.no_grad()
    def init_y(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_onehot = F.one_hot(t.clamp_min(0), self.k).float()
        return self.aux_mu(torch.cat([x, t_onehot], dim=-1))

    def contrastive_y_loss(
        self,
        x: torch.Tensor,
        y_pos: torch.Tensor,
        t: torch.Tensor,
        num_neg: int = 4,
        sigma: float = 0.1,
    ) -> torch.Tensor:
        noise = torch.randn_like(y_pos.unsqueeze(1).repeat(1, num_neg, 1)) * sigma
        y_neg = y_pos.unsqueeze(1) + noise
        e_pos = self.energy(x, y_pos).gather(1, t.view(-1, 1))
        e_neg = self.energy(
            x.unsqueeze(1).repeat(1, num_neg, 1).view(-1, x.size(-1)),
            y_neg.view(-1, y_pos.size(-1)),
        ).gather(1, t.unsqueeze(1).repeat(1, num_neg, 1).view(-1, 1))
        e_neg = e_neg.view(y_pos.size(0), num_neg)
        logits = torch.cat([-e_pos, -e_neg], dim=1)
        targets = torch.zeros(y_pos.size(0), dtype=torch.long, device=x.device)
        return F.cross_entropy(logits, targets)

    def energy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, y], dim=-1)
        base_E = self.energy_net(inp)
        reg = (y**2).mean(-1, keepdim=True) * self.beta_reg
        return base_E + reg

    # --------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        steps: int = 20,
        lr: float = 0.1,
        *,
        noise_std: float = 0.0,
        noise_decay: float = 1.0,
    ) -> torch.Tensor:
        """Approximate ``p(y|x,t)`` via gradient-based energy minimisation.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        t : torch.Tensor
            Treatment indicator.
        steps : int, optional
            Number of gradient updates.
        lr : float, optional
            Step size for gradient descent.
        noise_std : float, optional
            Unused but kept for backwards compatibility.
        noise_decay : float, optional
            Unused but kept for backwards compatibility.
        """

        y = self.init_y(x, t).detach().requires_grad_(True)
        opt = torch.optim.Adam([y], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            e = self.energy(x, y).gather(1, t.view(-1, 1)).sum()
            e.backward()
            opt.step()
            if y.grad is not None and y.grad.norm() < 1e-3:
                break
        return y.detach()

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_obs: torch.Tensor,
        *,
        w_lab: float = 1.0,
        w_ulb: float = 0.1,
        w_con: float = 1.0,
        alpha_aux: float = 0.1,
    ) -> torch.Tensor:
        """Compute the contrastive training loss.

        The energies are scaled by the temperature parameter before applying the
        labelled or marginal objectives.
        """

        energies = self.energy(x, y)  # (B,k)
        tau = self.log_tau.exp()
        energies = energies / tau
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
        loss_con = torch.tensor(0.0, device=x.device)
        loss_aux = torch.tensor(0.0, device=x.device)
        if labelled.any():
            loss_con = self.contrastive_y_loss(x[labelled], y[labelled], t_lab)
            loss_aux = F.mse_loss(self.init_y(x[labelled], t_lab), y[labelled])

        total = w_lab * loss_lab + w_ulb * loss_ulb + w_con * loss_con
        total = total + alpha_aux * loss_aux
        return total

    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return ``p(t|x,y)`` under the current energy function."""

        energies = self.energy(x, y)
        energies = energies / self.log_tau.exp()
        return F.softmax(-energies, dim=-1)

    @torch.no_grad()
    def predict_outcome(
        self, x: torch.Tensor, t: torch.Tensor, steps: int = 20, lr: float = 0.1
    ) -> torch.Tensor:
        """Predict the outcome by minimising the energy for the given treatment."""

        return self.forward(x, t, steps=steps, lr=lr)


__all__ = ["JointEBM"]
