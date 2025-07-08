from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable

from .registry import register_model
from .vime_self_pt import train_vime_s
from .vime_semi_pt import train_vime_sl


@register_model("vime")
class VIME_Model:
    """Wrapper that trains the two-stage VIME algorithm."""

    def __init__(
        self,
        p_m: float = 0.3,
        alpha: float = 2.0,
        K: int = 3,
        beta: float = 10.0,
        *,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        self.cfg = {"p_m": p_m, "alpha": alpha, "K": K, "beta": beta}
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation
        self.dropout = dropout
        self.norm_layer = norm_layer
        self.clf: nn.Module | None = None

    # --------------------------------------------------------------
    def fit(
        self,
        X_lab,
        y_lab,
        X_unlab,
        *,
        pre_epochs: int = 50,
        pre_bs: int = 256,
        sl_epochs: int = 200,
        sl_bs: int = 128,
    ) -> "VIME_Model":
        enc = train_vime_s(
            X_unlab,
            p_m=self.cfg["p_m"],
            alpha=self.cfg["alpha"],
            epochs=pre_epochs,
            bs=pre_bs,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            norm_layer=self.norm_layer,
        )
        self.clf = train_vime_sl(
            enc,
            X_lab,
            y_lab,
            X_unlab,
            K=self.cfg["K"],
            beta=self.cfg["beta"],
            p_m=self.cfg["p_m"],
            epochs=sl_epochs,
            bs=sl_bs,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            norm_layer=self.norm_layer,
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
