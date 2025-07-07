from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .vime_self_pt import Encoder, mask_corrupt


class Classifier(nn.Module):
    """Classifier head on top of a frozen VIME encoder."""

    def __init__(self, enc: Encoder, out_dim: int) -> None:
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(enc.net[-2].out_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))


def train_vime_sl(
    enc: Encoder,
    X_lab,
    y_lab,
    X_unlab,
    *,
    K: int = 3,
    beta: float = 10.0,
    epochs: int = 200,
    bs: int = 128,
) -> Classifier:
    """Semi-supervised fine-tuning of a VIME encoder."""

    Xl = torch.as_tensor(X_lab, dtype=torch.float32)
    yl = torch.as_tensor(y_lab, dtype=torch.float32)
    Xu = torch.as_tensor(X_unlab, dtype=torch.float32)
    clf = Classifier(enc, yl.shape[1])
    opt = optim.Adam(clf.parameters(), 3e-4)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    for _ in range(epochs):
        idx = torch.randint(0, len(Xl), (bs,))
        x_l, y_l = Xl[idx], yl[idx]
        idx_u = torch.randint(0, len(Xu), (bs,))
        x_u = Xu[idx_u]
        x_u_k = torch.stack([mask_corrupt(x_u, p_m=0.3)[1] for _ in range(K)])
        logits_l = clf(x_l)
        sup = ce(logits_l, y_l.argmax(1))
        logits_u = clf(x_u_k.view(-1, x_u.shape[1])).view(K, bs, -1)
        unsup = mse(logits_u.mean(0), logits_u).mean()
        loss = sup + beta * unsup
        opt.zero_grad()
        loss.backward()
        opt.step()
    return clf
