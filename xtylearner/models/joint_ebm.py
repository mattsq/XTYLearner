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

        self._cached_y: torch.Tensor | None = None

    def energy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, y], dim=-1)
        return self.energy_net(inp)

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
            Standard deviation of injected Gaussian noise for Langevin updates.
        noise_decay : float, optional
            Multiplicative factor applied to ``noise_std`` at each step.

        The final iterate is cached and reused on the next call when the batch
        shape matches, providing a simple form of amortised inference.
        """

        with torch.enable_grad():
            if self._cached_y is not None and self._cached_y.shape[0] == x.size(0):
                y = self._cached_y.clone().to(x.device).requires_grad_(True)
            else:
                y = torch.zeros(
                    x.size(0), self.d_y, device=x.device, requires_grad=True
                )
            for i in range(steps):
                e_all = self.energy_net(torch.cat([x, y], dim=-1))
                e = e_all.gather(1, t.view(-1, 1).clamp_min(0))
                grad = torch.autograd.grad(e.sum(), y, create_graph=False)[0]
                y = y - lr * grad
                if noise_std > 0:
                    y = y + torch.randn_like(y) * (noise_std * (noise_decay**i))
                y = y.detach().requires_grad_(True)
        self._cached_y = y.detach()
        return self._cached_y

    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute the contrastive training loss.

        The energies are scaled by the temperature parameter before applying the
        labelled or marginal objectives.
        """

        energies = self.energy_net(torch.cat([x, y], dim=-1))  # (B,k)
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
        return loss_lab + loss_ulb

    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return ``p(t|x,y)`` under the current energy function."""

        energies = self.energy_net(torch.cat([x, y], dim=-1))
        energies = energies / self.log_tau.exp()
        return F.softmax(-energies, dim=-1)

    @torch.no_grad()
    def predict_outcome(
        self, x: torch.Tensor, t: torch.Tensor, steps: int = 20, lr: float = 0.1
    ) -> torch.Tensor:
        """Predict the outcome by minimising the energy for the given treatment."""

        return self.forward(x, t, steps=steps, lr=lr)


__all__ = ["JointEBM"]
