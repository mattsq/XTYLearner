from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import register_model
from .layers import make_mlp
from .fixmatch_tabular import train_fixmatch


@register_model("fixmatch")
class FixMatch:
    """Simple FixMatch wrapper for semi-supervised classification."""

    def __init__(self, tau: float = 0.95, lambda_u: float = 1.0, mu: int = 7) -> None:
        self.cfg = {"tau": tau, "lambda_u": lambda_u, "mu": mu}
        self.net: torch.nn.Module | None = None

    # --------------------------------------------------------------
    def fit(self, X_lab, y_lab, X_unlab):
        Xl = torch.as_tensor(X_lab, dtype=torch.float32)
        yl = torch.as_tensor(y_lab, dtype=torch.long)
        Xu = torch.as_tensor(X_unlab, dtype=torch.float32)

        n_class = int(yl.max()) + 1
        self.net = make_mlp([Xl.size(1), 128, n_class]).to("cuda")
        lab_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xl, yl), batch_size=64, shuffle=True
        )
        unlab_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xu),
            batch_size=64 * self.cfg["mu"],
            shuffle=True,
        )

        train_fixmatch(
            self.net,
            lab_loader,
            unlab_loader,
            tau=self.cfg["tau"],
            lambda_u=self.cfg["lambda_u"],
        )
        self.net.eval()
        return self

    # --------------------------------------------------------------
    def predict_proba(self, X):
        X = torch.as_tensor(X, dtype=torch.float32).cuda()
        with torch.no_grad():
            out = self.net(X)
            return F.softmax(out, dim=1).cpu().numpy()

    # --------------------------------------------------------------
    def predict(self, X):
        return self.predict_proba(X).argmax(1)


