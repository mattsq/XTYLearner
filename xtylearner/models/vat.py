from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .layers import make_mlp
from .utils import ramp_up_sigmoid


def _l2_normalise(d: torch.Tensor) -> torch.Tensor:
    """Normalise a tensor batch-wise in L2 norm."""
    flat = d.view(d.size(0), -1)
    norm = torch.norm(flat, dim=1, keepdim=True) + 1e-8
    return d / norm.view(-1, *([1] * (d.dim() - 1)))


def vat_loss(model: nn.Module, x: torch.Tensor, xi: float = 1e-6, eps: float = 2.5, n_power: int = 1) -> torch.Tensor:
    """Virtual adversarial loss for a batch of inputs."""
    with torch.no_grad():
        pred = F.softmax(model(x), dim=1)

    d = torch.randn_like(x)
    for _ in range(n_power):
        d = xi * _l2_normalise(d)
        d.requires_grad_()
        pred_hat = model(x + d)
        adv_dist = F.kl_div(F.log_softmax(pred_hat, dim=1), pred, reduction="batchmean")
        grad = torch.autograd.grad(adv_dist, d)[0]
        d = grad.detach()

    r_adv = eps * _l2_normalise(d)
    pred_hat = model(x + r_adv)
    loss = F.kl_div(F.log_softmax(pred_hat, dim=1), pred, reduction="batchmean")
    return loss

@register_model("vat")
class VAT_Model:
    """Lightweight VAT wrapper for semi-supervised classification."""

    def __init__(self, eps: float = 2.5, xi: float = 1e-6, n_power: int = 1, lambda_max: float = 1.0, ramp: int = 30) -> None:
        self.cfg = {"eps": eps, "xi": xi, "n_power": n_power, "lambda_max": lambda_max, "ramp": ramp}
        self.net: nn.Module | None = None

    # --------------------------------------------------------------
    def fit(self, X_lab, y_lab, X_unlab, epochs: int = 200, bs: int = 256):
        Xl = torch.as_tensor(X_lab, dtype=torch.float32)
        yl = torch.as_tensor(y_lab, dtype=torch.long)
        Xu = torch.as_tensor(X_unlab, dtype=torch.float32)

        n_class = int(yl.max()) + 1
        self.net = make_mlp([Xl.size(1), 128, n_class])
        opt = torch.optim.AdamW(self.net.parameters(), 3e-4)
        ce = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            idx_l = torch.randint(0, len(Xl), (bs,))
            idx_u = torch.randint(0, len(Xu), (bs * 3,))
            x_l, y_l = Xl[idx_l], yl[idx_l]
            x_u = Xu[idx_u]

            logits = self.net(x_l)
            L_sup = ce(logits, y_l)
            L_vat = vat_loss(self.net, torch.cat([x_l, x_u]), xi=self.cfg["xi"], eps=self.cfg["eps"], n_power=self.cfg["n_power"])
            lam = ramp_up_sigmoid(epoch, self.cfg["ramp"], self.cfg["lambda_max"])
            loss = L_sup + lam * L_vat

            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    # --------------------------------------------------------------
    def predict_proba(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out = self.net(X)
            return F.softmax(out, dim=1).cpu().numpy()

    # --------------------------------------------------------------
    def predict(self, X):
        return self.predict_proba(X).argmax(1)


__all__ = ["VAT_Model", "vat_loss"]
