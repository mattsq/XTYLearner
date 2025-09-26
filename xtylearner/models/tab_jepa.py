# Bacharach et al. "I-JEPA: Moving Self-Supervised Learning Beyond Pixels" • ICCV 2023

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from .layers import ColumnEmbedder, apply_column_mask, make_mlp
from .heads import LowRankDiagHead
from ..losses import nll_lowrank_diag
from ..training.metrics import mse_loss

from .registry import register_model


@register_model("tab_jepa")
class TabJEPA(nn.Module):
    """Tabular Joint-Embedding Predictive Architecture.

    Pre-trains by masking column tokens of ``(X, T, Y)`` and predicting the
    latent representation of the unmasked sequence.
    During fine-tuning ``forward`` returns outcome predictions from the
    aggregated representation.
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int | None = None,
        *,
        d_embed: int = 64,
        depth: int = 4,
        nhead: int = 8,
        dim_feedforward: int | None = None,
        mask_ratio: float = 0.4,
        momentum: float = 0.996,
        lowrank_head: bool = False,
        rank: int = 4,
        λ_jepa: float = 1.0,
        λ_sup: float = 1.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.mask_ratio = mask_ratio
        self.momentum = momentum
        self.lowrank_head = lowrank_head
        self.λ_jepa = λ_jepa
        self.λ_sup = λ_sup

        # column tokeniser
        self.embed = ColumnEmbedder(d_x, d_y, k, d_embed)

        # encoders
        if dim_feedforward is None:
            dim_feedforward = 4 * d_embed
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_embed, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=depth,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # predictor head
        self.predictor = make_mlp([d_embed, d_embed, d_embed])

        # downstream Y head
        if lowrank_head:
            self.Y_head = LowRankDiagHead(d_embed, d_y, rank)
        else:
            self.Y_head = nn.Linear(d_embed, d_y)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _momentum_update(self) -> None:
        """Update target encoder with exponential moving average."""

        for q, k in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            q.data.mul_(self.momentum).add_(k.data, alpha=1.0 - self.momentum)

    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor, T: torch.Tensor):
        """Predict outcomes from covariates and treatments."""

        if self.k is None:
            T_tok = T.unsqueeze(-1).float()
        else:
            T_tok = one_hot(T.to(torch.long), self.k).float()
        tokens = self.embed(X, T_tok)
        h = self.encoder(tokens).mean(1)
        if self.lowrank_head:
            mu, F, sigma2 = self.Y_head(h)
            return mu, F, sigma2
        else:
            return self.Y_head(h)

    @torch.no_grad()
    def predict_outcome(self, X: torch.Tensor, T: torch.Tensor):
        out = self.forward(X, T)
        return out[0] if self.lowrank_head else out

    # ------------------------------------------------------------------
    def loss(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        T_obs: torch.Tensor,
        *,
        λ_jepa: Optional[float] = None,
        λ_sup: Optional[float] = None,
    ) -> torch.Tensor:
        """Blend JEPA pre-text loss with supervised outcome loss."""

        if λ_jepa is None:
            λ_jepa = self.λ_jepa
        if λ_sup is None:
            λ_sup = self.λ_sup

        if self.k is None:
            T_tok = T_obs.unsqueeze(-1).float()
        else:
            T_tok = one_hot(T_obs.clamp(min=0).to(torch.long), self.k).float()

        # construct token sequence
        full_tokens = self.embed(X, T_tok)

        # apply column mask
        masked_tokens, mask = apply_column_mask(full_tokens, self.mask_ratio)

        # encoders
        h_ctx = self.encoder(masked_tokens)
        with torch.no_grad():
            h_tgt = self.target_encoder(full_tokens)

        # JEPA loss
        pred = self.predictor(h_ctx[mask])
        target = h_tgt[mask]
        L_jepa = (pred - target).pow(2).mean()

        self._momentum_update()

        # supervised loss on labelled rows
        labelled = T_obs >= 0
        if labelled.any():
            Y_hat = self.predict_outcome(X[labelled], T_obs[labelled])
            if self.lowrank_head:
                mu, F, sigma2 = Y_hat
                L_sup = nll_lowrank_diag(Y[labelled], mu, F, sigma2).mean()
            else:
                L_sup = mse_loss(Y_hat, Y[labelled])
        else:
            L_sup = 0.0

        return λ_jepa * L_jepa + λ_sup * L_sup

    # ------------------------------------------------------------------
    def predict_treatment_proba(self, X: torch.Tensor, Y: torch.Tensor):
        if self.k is None:
            return torch.full((X.size(0),), float("nan"), device=X.device)
        return torch.full((X.size(0), self.k), 1 / self.k, device=X.device)


__all__ = ["TabJEPA"]
