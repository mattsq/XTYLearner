from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from .registry import register_model
from .layers import make_mlp
from .vime_self_pt import Encoder, Decoder, mask_corrupt


@register_model("vime")
class VIME(nn.Module):
    """Two-stage VIME model with trainer-friendly interface."""

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int = 2,
        *,
        p_m: float = 0.3,
        alpha: float = 2.0,
        K: int = 3,
        beta: float = 10.0,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.p_m = p_m
        self.alpha = alpha
        self.K = K
        self.beta = beta
        self.k = k
        self.d_y = d_y

        self.encoder = Encoder(
            d_x,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.decoder = Decoder(
            d_x,
            self.encoder.out_dim,
            hidden_dims=(),
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.classifier = make_mlp(
            [self.encoder.out_dim, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = self.encoder(x)
        return self.classifier(h)

    # --------------------------------------------------------------
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, _t_obs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Combined loss for the two-stage VIME algorithm."""

        if y.dim() > 1:
            y = y.argmax(1)
        y = y.to(torch.long)

        # ---- stage 1 : self-supervised pre-training -----------------
        m, xt = mask_corrupt(x, self.p_m)
        h_pt = self.encoder(xt)
        m_hat, x_hat = self.decoder(h_pt)
        loss_pre = F.binary_cross_entropy(m_hat, m) + self.alpha * F.mse_loss(x_hat, x)

        # ---- stage 2 : semi-supervised classification ----------------
        logits = self.forward(x)
        labelled = y >= 0
        loss_sup = (
            F.cross_entropy(logits[labelled], y[labelled])
            if labelled.any()
            else torch.tensor(0.0, device=x.device)
        )

        if (~labelled).any():
            x_u = x[~labelled]
            x_u_k = torch.stack(
                [mask_corrupt(x_u, p_m=self.p_m)[1] for _ in range(self.K)]
            )
            logits_u = self.forward(x_u_k.view(-1, x_u.size(1))).view(
                self.K, x_u.size(0), -1
            )
            loss_unsup = F.mse_loss(
                logits_u.mean(0, keepdim=True).expand_as(logits_u), logits_u
            )
        else:
            loss_unsup = torch.tensor(0.0, device=x.device)

        # total --------------------------------------------------------
        loss = loss_pre + loss_sup + self.beta * loss_unsup
        return {
            "loss": loss,
            "pretrain": loss_pre,
            "supervised": loss_sup,
            "unsup": loss_unsup,
        }

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor | list) -> torch.Tensor:
        X_t = torch.as_tensor(
            X, dtype=torch.float32, device=next(self.parameters()).device
        )
        logits = self.forward(X_t)
        return logits.softmax(dim=-1).cpu()

    # --------------------------------------------------------------
    def predict_treatment_proba(
        self, x: torch.Tensor, _y: torch.Tensor
    ) -> torch.Tensor:
        """Alias of :meth:`predict_proba` for API compatibility."""

        return self.predict_proba(x)

    # --------------------------------------------------------------
    def predict(self, X: torch.Tensor | list) -> torch.Tensor:
        return self.predict_proba(X).argmax(dim=1)


__all__ = ["VIME"]
