from __future__ import annotations

from typing import Iterable, Optional

import torch

from .base_trainer import BaseTrainer
from .supervised import SupervisedTrainer
from .adversarial import AdversarialTrainer
from .generative import GenerativeTrainer
from .diffusion import DiffusionTrainer
from .em import ArrayTrainer
from .logger import TrainerLogger


class Trainer:
    """Factory wrapper selecting an appropriate concrete trainer."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: (
            torch.optim.Optimizer | tuple[torch.optim.Optimizer, torch.optim.Optimizer]
        ),
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        device: str = "cpu",
        logger: Optional[TrainerLogger] = None,
        scheduler: (
            torch.optim.lr_scheduler._LRScheduler
            | tuple[
                torch.optim.lr_scheduler._LRScheduler,
                torch.optim.lr_scheduler._LRScheduler,
            ]
            | None
        ) = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        """Instantiate an appropriate trainer for ``model`` and delegate calls."""

        trainer_cls = self._select_trainer(model)
        if trainer_cls is AdversarialTrainer:
            if not isinstance(optimizer, (tuple, list)) or len(optimizer) != 2:
                raise ValueError("AdversarialTrainer requires two optimizers")
            optim_G, optim_D = optimizer
            self._trainer = trainer_cls(
                model,
                optim_G,
                optim_D,
                train_loader,
                val_loader,
                device,
                logger,
                scheduler,
                grad_clip_norm,
            )
        else:
            self._trainer = trainer_cls(
                model,
                optimizer,  # type: ignore[arg-type]
                train_loader,
                val_loader,
                device,
                logger,
                scheduler,
                grad_clip_norm,
            )

    # ------------------------------------------------------------------
    def _select_trainer(self, model: torch.nn.Module) -> type[BaseTrainer]:
        """Pick a trainer subclass based on the interfaces exposed by ``model``."""
        if hasattr(model, "loss_G") and hasattr(model, "loss_D"):
            return AdversarialTrainer
        if hasattr(model, "elbo"):
            return GenerativeTrainer
        if hasattr(model, "sample") or hasattr(model, "paired_sample"):
            return DiffusionTrainer
        if hasattr(model, "predict_outcome") and not hasattr(model, "train"):
            return ArrayTrainer
        if (
            hasattr(model, "fit")
            and not hasattr(model, "loss")
            and not hasattr(model, "train")
        ):
            return ArrayTrainer
        return SupervisedTrainer

    # ------------------------------------------------------------------
    def fit(self, num_epochs: int) -> None:
        """Train the wrapped model for ``num_epochs`` epochs."""
        self._trainer.fit(num_epochs)

    def evaluate(self, data_loader: Iterable) -> float:
        """Return the primary metric on ``data_loader``."""
        return self._trainer.evaluate(data_loader)

    def predict(self, *args, **kwargs):
        """Forward to the underlying trainer's ``predict`` method."""
        return self._trainer.predict(*args, **kwargs)

    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Delegated call to ``predict_treatment_proba`` of the inner trainer."""
        return self._trainer.predict_treatment_proba(x, y)

    # Expose attributes of the underlying trainer/model
    def __getattr__(self, name):
        return getattr(self._trainer, name)


__all__ = ["Trainer"]
