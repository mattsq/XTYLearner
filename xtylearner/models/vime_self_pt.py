from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable

from .layers import make_mlp


class Encoder(nn.Module):
    """MLP encoder used for VIME pre-training."""

    def __init__(
        self,
        d: int,
        *,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            [d, *hidden_dims],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.out_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Head decoding the mask and feature reconstruction."""

    def __init__(
        self,
        d: int,
        rep_dim: int,
        *,
        hidden_dims: tuple[int, ...] | list[int] = (),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        dims = [rep_dim, *hidden_dims, d]
        self.mask = make_mlp(
            dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.recon = make_mlp(
            dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.mask(h)), self.recon(h)


def mask_corrupt(X: torch.Tensor, p_m: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply column-wise masking and imputation."""

    m = torch.bernoulli(torch.full_like(X, p_m))
    X_tilde = X.clone()
    for j in range(X.shape[1]):
        col = X[:, j]
        X_tilde[:, j] = torch.where(
            m[:, j] == 1,
            col,
            col[torch.randint(0, len(col), (len(col),))],
        )
    return m, X_tilde


def train_vime_s(
    X_unlab: torch.Tensor | list,
    *,
    p_m: float = 0.3,
    alpha: float = 2.0,
    epochs: int = 50,
    bs: int = 256,
    hidden_dims: tuple[int, ...] | list[int] = (128, 128),
    activation: type[nn.Module] = nn.ReLU,
    dropout: float | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    lr: float = 3e-4,
) -> Encoder:
    """Self-supervised pre-training stage for VIME."""

    X_unlab = torch.as_tensor(X_unlab, dtype=torch.float32)
    enc = Encoder(
        X_unlab.shape[1],
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
        norm_layer=norm_layer,
    )
    dec = Decoder(
        X_unlab.shape[1],
        enc.out_dim,
        hidden_dims=(),
        activation=activation,
        dropout=dropout,
        norm_layer=norm_layer,
    )
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr)
    loader = DataLoader(TensorDataset(X_unlab), bs, shuffle=True)
    bce, mse = nn.BCELoss(), nn.MSELoss()
    for _ in range(epochs):
        for (x,) in loader:
            m, xt = mask_corrupt(x, p_m)
            h = enc(xt)
            m_hat, x_hat = dec(h)
            loss = bce(m_hat, m) + alpha * mse(x_hat, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return enc
