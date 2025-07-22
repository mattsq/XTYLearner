from __future__ import annotations

from typing import Iterable, Optional

import torch

from .base_trainer import BaseTrainer
from .logger import TrainerLogger


class AdversarialTrainer(BaseTrainer):
    """Trainer for models with ``loss_G`` and ``loss_D`` methods."""

    def __init__(
        self,
        model: torch.nn.Module,
        optim_G: torch.optim.Optimizer,
        optim_D: torch.optim.Optimizer,
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
        """Create a trainer for adversarial models.

        Parameters
        ----------
        model:
            Model exposing ``loss_G`` and ``loss_D`` methods.
        optim_G:
            Optimiser updating the generator parameters.
        optim_D:
            Optimiser updating the discriminator parameters.
        train_loader:
            Iterable of training batches.
        val_loader:
            Optional iterable of validation batches.
        device:
            Device identifier for training.
        logger:
            Optional :class:`TrainerLogger` implementation.
        scheduler:
            Learning rate scheduler or pair of schedulers for G/D.
        grad_clip_norm:
            If set, gradients are clipped to this norm.
        """
        super().__init__(
            model,
            optim_G,
            train_loader,
            val_loader,
            device,
            logger,
            scheduler,
            grad_clip_norm,
        )
        self.optim_G = optim_G
        self.optim_D = optim_D

    # --------------------------------------------------------------
    def step(self, batch: Iterable[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Perform one update of generator and discriminator.

        Parameters
        ----------
        batch:
            Iterable yielding ``(X, Y, T)`` tensors on the correct device.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing ``loss_G`` and ``loss_D`` values.
        """
        x, y, t = self._extract_batch(batch)

        if not torch.is_grad_enabled():
            loss = self.model.loss_G(x, y, t)
            loss.update(self.model.loss_D(x, y, t))
            return loss

        self.optim_G.zero_grad()
        loss_dict = self.model.loss_G(x, y, t)
        loss_dict["loss_G"].backward()
        self._clip_grads()
        self.optim_G.step()

        self.optim_D.zero_grad()
        loss_dict.update(self.model.loss_D(x, y, t))
        loss_dict["loss_D"].backward()
        self._clip_grads()
        self.optim_D.step()
        return loss_dict

    # --------------------------------------------------------------
    def fit(self, num_epochs: int) -> None:
        """Alternately update generator and discriminator.

        Parameters
        ----------
        num_epochs:
            Number of training epochs to run.
        """
        for epoch in range(num_epochs):
            self.model.train()
            num_batches = len(self.train_loader)
            if self.logger:
                self.logger.start_epoch(epoch + 1, num_batches)
            for batch_idx, batch in enumerate(self.train_loader):
                X, Y, T_obs = self._extract_batch(batch)
                losses = self.step(batch)
                if self.logger:
                    metrics = dict(self._metrics_from_loss(losses))
                    metrics.update(self._treatment_metrics(X, Y, T_obs))
                    metrics.update(self._outcome_metrics(X, Y, T_obs))
                    self.logger.log_step(epoch + 1, batch_idx, num_batches, metrics)
            if self.scheduler is not None:
                if isinstance(self.scheduler, (tuple, list)):
                    for sched in self.scheduler:
                        if sched is not None:
                            sched.step()
                else:
                    self.scheduler.step()
            if self.logger and self.val_loader is not None:
                val_metrics = self._eval_metrics(self.val_loader)
                self.logger.log_validation(epoch + 1, val_metrics)
            if self.logger:
                self.logger.end_epoch(epoch + 1)

    # --------------------------------------------------------------
    def evaluate(self, data_loader: Iterable) -> Mapping[str, float]:
        """Return metrics on ``data_loader``.

        Parameters
        ----------
        data_loader:
            Iterable providing evaluation batches.

        Returns
        -------
        Mapping[str, float]
            Dictionary with loss, treatment accuracy and RMSE metrics.
        """
        metrics = self._eval_metrics(data_loader)
        loss_val = metrics.get("loss_G", next(iter(metrics.values()), 0.0))
        return {
            "loss": float(loss_val),
            "treatment accuracy": float(metrics.get("accuracy", 0.0)),
            "outcome rmse": float(metrics.get("rmse", 0.0)),
            "outcome rmse labelled": float(metrics.get("rmse_labelled", 0.0)),
            "outcome rmse unlabelled": float(metrics.get("rmse_unlabelled", 0.0)),
        }

    def predict(self, *inputs: torch.Tensor):
        """Return model outputs for ``inputs`` without gradient tracking.

        Parameters
        ----------
        *inputs:
            Input tensors forwarded to the model.

        Returns
        -------
        torch.Tensor | Any
            Predictions produced by the underlying model.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = [i.to(self.device) for i in inputs]
            if hasattr(self.model, "predict"):
                return self.model.predict(*inputs)
            return self.model(*inputs)


__all__ = ["AdversarialTrainer"]
