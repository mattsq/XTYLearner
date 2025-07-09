"""Two-stage deconfounder / causal-factor model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import VAE_T, OutcomeNet
from .registry import register_model


def _hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute an unbiased HSIC estimate with RBF kernels."""
    n = x.size(0)
    xx = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(-1)
    yy = (y.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)
    kx = torch.exp(-xx / x.size(1))
    ky = torch.exp(-yy / y.size(1))
    h = torch.eye(n, device=x.device) - torch.full((n, n), 1.0 / n, device=x.device)
    kxc = h @ kx @ h
    kyc = h @ ky @ h
    return (kxc * kyc).sum() / (n - 1) ** 2


@register_model("deconfounder_cfm")
class DeconfounderCFM(nn.Module):
    """Two-stage causal-factor model with substitute confounder."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k_t: int,
        d_z: int = 16,
        hidden: int = 64,
        pretrain_epochs: int = 50,
        ppc_freq: int = 25,
    ) -> None:
        super().__init__()
        self.vae_t = VAE_T(k_t, d_z, hidden)
        self.out_net = OutcomeNet(d_x + d_z + k_t, d_y, hidden)
        self.register_buffer("epoch", torch.zeros(1, dtype=torch.long))
        self.pretrain_epochs = pretrain_epochs
        self.ppc_freq = ppc_freq

    # --------------------------------------------------------------
    def loss(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.epoch.item() < self.pretrain_epochs:
            return self.vae_t.elbo(t)
        elbo_t = self.vae_t.elbo(t)
        z = self.vae_t.encode(t, sample=False).detach()
        inp = torch.cat([x, torch.nan_to_num(t, 0.0), z], 1)
        pred = self.out_net(inp)
        loss_y = F.mse_loss(pred, y)
        return elbo_t + loss_y

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = self.vae_t.encode(t, sample=False)
        inp = torch.cat([x, torch.nan_to_num(t, 0.0), z], 1)
        return self.out_net(inp)

    @torch.no_grad()
    def predict_ite(
        self, x: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor
    ) -> torch.Tensor:
        y0 = self.predict_outcome(x, t0)
        y1 = self.predict_outcome(x, t1)
        return y1 - y0

    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Not implemented since the model does not learn ``p(t|x,y)``."""

        raise NotImplementedError(
            "DeconfounderCFM does not model p(t|x,y); treatment probabilities "
            "are unavailable."
        )

    def on_epoch_end(self, t: torch.Tensor | None = None) -> None:
        self.epoch += 1
        if t is not None and self.epoch.item() % self.ppc_freq == 0:
            z = self.vae_t.encode(t, sample=False)
            ppc = _hsic(torch.nan_to_num(t, 0.0), z)
            if hasattr(self, "logger"):
                self.logger.log_scalar("ppc_hsic", float(ppc))


__all__ = ["DeconfounderCFM"]
