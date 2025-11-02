"""Lightweight estimator for label observability under MNAR selection bias."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _LabelPropensityNet(nn.Module):
    r"""Two-layer MLP estimating :math:`P(L=1 \mid X=x)`.

    The network is intentionally small to keep training inexpensive during
    active learning rounds.  It acts as a proxy for the probability that a
    sample yields a reliable label.  This follows the idea described in the
    DebiasedCoverage acquisition strategy: distinguish covariate regions where
    labels are routinely observable from those that are effectively unreachable.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        hidden_dim = max(0, int(hidden_dim))
        if hidden_dim == 0:
            self._net = nn.Linear(in_dim, 1)
        else:
            self._net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x))


@dataclass
class LabelPropensityTrainingConfig:
    """Configuration for the label propensity auxiliary model."""

    max_epochs: int = 30
    batch_size: int = 256
    lr: float = 5e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64


def _build_dataset(
    labeled: TensorDataset,
    unlabeled: TensorDataset,
) -> tuple[TensorDataset | None, int]:
    """Construct a binary dataset distinguishing labelled from unlabelled rows."""

    if len(labeled) == 0 or len(unlabeled) == 0:
        return None, 0

    x_pos = labeled.tensors[0]
    x_neg = unlabeled.tensors[0]
    if x_pos.numel() == 0 or x_neg.numel() == 0:
        return None, 0

    x = torch.cat([x_pos, x_neg], dim=0).float()
    y = torch.cat(
        [
            torch.ones(len(x_pos), dtype=torch.float32, device=x_pos.device),
            torch.zeros(len(x_neg), dtype=torch.float32, device=x_neg.device),
        ]
    )
    dataset = TensorDataset(x, y)
    return dataset, x_pos.size(1)


def train_label_propensity_model(
    labeled: TensorDataset,
    unlabeled: TensorDataset,
    *,
    device: torch.device | str = "cpu",
    config: LabelPropensityTrainingConfig | None = None,
) -> _LabelPropensityNet | None:
    r"""Fit a small classifier estimating :math:`P(L=1 \mid X=x)`.

    Parameters
    ----------
    labeled:
        Dataset of currently labelled rows.  These are treated as positive
        examples of *observable* labels.
    unlabeled:
        Dataset of rows without labels.  They act as weak negatives – following
        the heuristic described in the DebiasedCoverage strategy, we assume that
        if a point has remained unlabeled so far it is *less likely* to be
        reliably labelable.  Future improvements may distinguish between
        "unlabeled yet" and "attempted but failed".
    device:
        Device used for training the auxiliary model.
    config:
        Optional hyper-parameters overriding the defaults.

    Returns
    -------
    _LabelPropensityNet | None
        Trained network or ``None`` when insufficient data is available.
    """

    dataset, d_x = _build_dataset(labeled, unlabeled)
    if dataset is None:
        logger.debug("Skipping label propensity training due to insufficient data")
        return None

    cfg = config or LabelPropensityTrainingConfig()
    loader = DataLoader(
        dataset,
        batch_size=min(cfg.batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
    )

    model = _LabelPropensityNet(d_x, hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pos_count = float(len(labeled))
    neg_count = float(len(unlabeled))
    if pos_count == 0 or neg_count == 0:
        return None
    pos_weight_value = neg_count / max(pos_count, 1.0)
    pos_weight = torch.tensor(pos_weight_value, device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(cfg.max_epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model.logits(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


__all__ = [
    "LabelPropensityTrainingConfig",
    "train_label_propensity_model",
]
