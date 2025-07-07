from __future__ import annotations

import torch

from .registry import register_model
from .vime_self_pt import train_vime_s
from .vime_semi_pt import train_vime_sl


@register_model("vime")
class VIME_Model:
    """Wrapper that trains the two-stage VIME algorithm."""

    def __init__(self, p_m: float = 0.3, alpha: float = 2.0, K: int = 3, beta: float = 10.0) -> None:
        self.cfg = {"p_m": p_m, "alpha": alpha, "K": K, "beta": beta}
        self.clf = None

    # --------------------------------------------------------------
    def fit(self, X_lab, y_lab, X_unlab):
        enc = train_vime_s(X_unlab, p_m=self.cfg["p_m"], alpha=self.cfg["alpha"])
        self.clf = train_vime_sl(
            enc,
            X_lab,
            y_lab,
            X_unlab,
            K=self.cfg["K"],
            beta=self.cfg["beta"],
        )
        return self

    # --------------------------------------------------------------
    def predict_proba(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out = self.clf(X)
            return torch.softmax(out, -1).cpu().numpy()

    # --------------------------------------------------------------
    def predict(self, X):
        return self.predict_proba(X).argmax(1)
