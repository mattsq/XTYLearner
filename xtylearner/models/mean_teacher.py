from __future__ import annotations

from copy import deepcopy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


@register_model("mean_teacher")
class MeanTeacher(nn.Module):
    """Simple mean teacher wrapper for semi-supervised classification."""

    def __init__(
        self,
        base_net_fn,
        num_classes: int,
        ema_decay: float = 0.99,
        ramp_up: int = 40,
        cons_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.student = base_net_fn(num_classes)
        self.teacher = deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.ema_decay = ema_decay
        self.ramp_up = ramp_up
        self.cons_max = cons_max
        self.step = 0  # global step counter

    # --------------------------------------------------------------
    def _consistency_weight(self, epoch: int) -> torch.Tensor:
        t = min(epoch / self.ramp_up, 1.0)
        return self.cons_max * math.exp(-5 * (1 - t) ** 2)

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, teacher: bool = False) -> torch.Tensor:
        net = self.teacher if teacher else self.student
        return net(x)

    # --------------------------------------------------------------
    def update_teacher(self) -> None:
        alpha = self.ema_decay
        for theta_s, theta_t in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            theta_t.data.mul_(alpha).add_(theta_s.data, alpha=1 - alpha)

    # --------------------------------------------------------------
    def fit(
        self,
        X_lab: torch.Tensor,
        y_lab: torch.Tensor,
        X_unlab: torch.Tensor,
        *,
        epochs: int = 200,
        batch_size: int = 64,
        mu: int = 3,
        lr: float = 3e-4,
    ) -> "MeanTeacher":
        """Train the student network with the Mean Teacher algorithm."""

        device = X_lab.device if X_lab.is_cuda else "cuda" if torch.cuda.is_available() else "cpu"
        self.student.to(device)
        self.teacher.to(device)

        lab_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_lab, y_lab), batch_size=batch_size, shuffle=True
        )
        unlab_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_unlab), batch_size=batch_size * mu, shuffle=True
        )

        opt = torch.optim.AdamW(self.student.parameters(), lr)

        for epoch in range(epochs):
            it_unlab = iter(unlab_loader)
            for x_l, y_l in lab_loader:
                try:
                    x_u = next(it_unlab)[0]
                except StopIteration:
                    it_unlab = iter(unlab_loader)
                    x_u = next(it_unlab)[0]

                x_l = x_l.to(device)
                y_l = y_l.to(device)
                x_u = x_u.to(device)

                def noise(inp: torch.Tensor) -> torch.Tensor:
                    std = inp.std(0, keepdim=True) + 1e-6
                    return inp + 0.05 * std * torch.randn_like(inp)

                x_lab_s = noise(x_l)
                x_unlab_s = noise(x_u)
                x_unlab_t = noise(x_u)

                logits_lab = self.student(x_lab_s)
                logits_s = self.student(x_unlab_s)
                with torch.no_grad():
                    logits_t = self.teacher(x_unlab_t)

                L_sup = F.cross_entropy(logits_lab, y_l)
                L_cons = F.mse_loss(logits_s.softmax(dim=-1), logits_t.softmax(dim=-1))
                lam = self._consistency_weight(epoch)
                loss = L_sup + lam * L_cons

                opt.zero_grad()
                loss.backward()
                opt.step()
                self.update_teacher()

        self.eval()
        return self

    # --------------------------------------------------------------
    def predict_proba(self, X: torch.Tensor, *, teacher: bool = True):
        """Return class probabilities from the teacher or student network."""

        net = self.teacher if teacher else self.student
        net.eval()
        with torch.no_grad():
            X = X.to(next(net.parameters()).device)
            logits = net(X)
            return logits.softmax(dim=-1).cpu().numpy()

    # --------------------------------------------------------------
    def predict(self, X: torch.Tensor, *, teacher: bool = True):
        return self.predict_proba(X, teacher=teacher).argmax(1)
