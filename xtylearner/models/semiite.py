from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model


@register_model("semiite")
class SemiITE(nn.Module):
    """Co-training network for semi-supervised treatment effect estimation."""

    def __init__(
        self,
        d_x: int,
        d_y: int = 1,
        k: int = 2,
        *,
        rep_dim: int = 128,
        lambda_u: float = 0.5,
        q_pseudo: int = 32,
        mmd_beta: float = 1e-2,
    ) -> None:
        super().__init__()
        if k != 2:
            raise ValueError("SemiITE currently supports binary treatment (k=2) only")
        self.k = k
        self.lambda_u = lambda_u
        self.q_pseudo = q_pseudo
        self.mmd_beta = mmd_beta

        self.enc = make_mlp([d_x, 256, rep_dim])
        self.prop = nn.Linear(rep_dim, k)
        self.outcome = nn.ModuleList(
            [make_mlp([rep_dim + k, 128, d_y]) for _ in range(3)]
        )

    # --------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        t1h = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome[0](torch.cat([z, t1h], dim=-1))

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        z = self.encode(x)
        return self.prop(z).softmax(dim=-1)

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
        t1h = F.one_hot(t.to(torch.long), self.k).float()
        return self.outcome[0](torch.cat([z, t1h], dim=-1))

    # --------------------------------------------------------------
    def compute_mmd(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        vals = t.unique()
        if vals.numel() < 2:
            return torch.tensor(0.0, device=z.device)
        means = [z[t == v].mean(0) for v in vals]
        mmd = 0.0
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                mmd += (means[i] - means[j]).pow(2).mean()
        return mmd / (len(means) * (len(means) - 1) / 2)


__all__ = ["SemiITE"]
