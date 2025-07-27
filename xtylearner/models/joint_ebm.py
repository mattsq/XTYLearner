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
from .utils import centre_per_row


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
        ulb_weight: float = 0.1,
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
        self._base_ulb_w = ulb_weight
        self._ulb_w = ulb_weight
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

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        """Update curriculum weight for unlabelled loss."""

        ramp = 0.3 * total_epochs
        self._ulb_w = self._base_ulb_w * min(1.0, epoch / ramp)

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
        l2 = (y ** 2).mean(-1, keepdim=True)
        if torch.is_grad_enabled() and y.requires_grad:
            (grad_y,) = torch.autograd.grad(l2.sum(), y, create_graph=True)
            grad_pen = (grad_y ** 2).mean()
        else:
            grad_pen = torch.tensor(0.0, device=y.device)
        return base_E + self.beta_reg * l2 + 1e-4 * grad_pen

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

        y = self.init_y(x, t)
        for _ in range(steps):
            y = y.detach().requires_grad_(True)
            e = self.energy(x, y).gather(1, t.view(-1, 1)).sum()
            (grad,) = torch.autograd.grad(e, y)
            y = (y - lr * grad).clamp(-10.0, 10.0)
            if grad.norm() < 1e-3:
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
        """Compute the contrastive training loss."""

        E = centre_per_row(self.energy(x, y)) / self.log_tau.exp()
        labelled = t_obs >= 0

        loss_lab = (
            F.cross_entropy(-E[labelled], t_obs[labelled]) if labelled.any() else 0.0
        )

        unlab_idx = ~labelled
        if unlab_idx.any():
            probs_ulb = F.softmax(-E[unlab_idx], dim=-1)
            target = probs_ulb.new_full(probs_ulb.shape, 1.0 / self.k)
            loss_ulb = F.kl_div(probs_ulb.log(), target, reduction="batchmean")
        else:
            loss_ulb = 0.0

        if labelled.any():
            loss_con = self.contrastive_y_loss(x[labelled], y[labelled], t_obs[labelled])
            loss_aux = F.mse_loss(self.init_y(x[labelled], t_obs[labelled]), y[labelled])
        else:
            loss_con = 0.0
            loss_aux = 0.0

        return (
            w_lab * loss_lab
            + self._ulb_w * loss_ulb
            + w_con * loss_con
            + alpha_aux * loss_aux
        )

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

        with torch.enable_grad():
            return self.forward(x, t, steps=steps, lr=lr)

    @torch.no_grad()
    def predict_marginal_y(
        self, x: torch.Tensor, y_obs: torch.Tensor, steps: int = 20, lr: float = 0.1
    ) -> torch.Tensor:
        """Predict outcome marginalising over unknown treatment."""

        p_t = self.predict_treatment_proba(x, y_obs)
        y_t = torch.stack(
            [
                self.predict_outcome(x, torch.full_like(y_obs[:, :1], t), steps, lr)
                for t in range(self.k)
            ],
            dim=-2,
        )
        return (p_t.unsqueeze(-1) * y_t).sum(dim=-2)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, t: torch.Tensor, steps: int = 20, lr: float = 0.1
    ) -> torch.Tensor:
        """Alias of :meth:`predict_outcome` used by :class:`Trainer`."""

        return self.predict_outcome(x, t, steps=steps, lr=lr)


__all__ = ["JointEBM"]
