from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Mapping

import torch


from .logger import TrainerLogger


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        device: str = "cpu",
        logger: Optional[TrainerLogger] = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger

    @abstractmethod
    def fit(self, num_epochs: int) -> None:
        """Train the model for ``num_epochs`` epochs."""

    @abstractmethod
    def evaluate(self, data_loader: Iterable) -> float:
        """Return a scalar loss/metric evaluated on ``data_loader``."""

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Return model predictions for the supplied inputs."""

    def _metrics_from_loss(
        self, loss: torch.Tensor | Mapping[str, float]
    ) -> Mapping[str, float]:
        if isinstance(loss, torch.Tensor):
            return {"loss": float(loss.item())}
        if isinstance(loss, Mapping):
            return {k: float(v) for k, v in loss.items()}
        return {"loss": float(loss)}
