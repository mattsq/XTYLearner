from __future__ import annotations

from typing import Iterable, Optional

import torch

from .base_trainer import BaseTrainer
from .supervised import SupervisedTrainer
from .generative import GenerativeTrainer
from .diffusion import DiffusionTrainer
from .logger import TrainerLogger


class Trainer:
    """High level trainer that dispatches to the appropriate implementation."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        device: str = "cpu",
        logger: Optional[TrainerLogger] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        trainer_cls = self._select_trainer(model)
        self._trainer: BaseTrainer = trainer_cls(
            model,
            optimizer,
            train_loader,
            val_loader,
            device,
            logger,
            scheduler,
            grad_clip_norm,
        )

    # ------------------------------------------------------------------
    def _select_trainer(self, model: torch.nn.Module) -> type[BaseTrainer]:
        if hasattr(model, "elbo"):
            return GenerativeTrainer
        if hasattr(model, "sample") or hasattr(model, "paired_sample"):
            return DiffusionTrainer
        return SupervisedTrainer

    # ------------------------------------------------------------------
    def fit(self, num_epochs: int) -> None:
        self._trainer.fit(num_epochs)

    def evaluate(self, data_loader: Iterable) -> float:
        return self._trainer.evaluate(data_loader)

    def predict(self, *args, **kwargs):
        return self._trainer.predict(*args, **kwargs)

    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._trainer.predict_treatment_proba(x, y)

    # Expose attributes of the underlying trainer/model
    def __getattr__(self, name):
        return getattr(self._trainer, name)


__all__ = ["Trainer"]
