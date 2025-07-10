from __future__ import annotations

from typing import Iterable, Mapping

import torch

from .base_trainer import BaseTrainer


class SupervisedTrainer(BaseTrainer):
    """Generic trainer for fully observed models with a ``loss`` method."""

    def step(self, batch: Iterable[torch.Tensor]):
        """Return the model loss for ``batch`` moved to the current device."""
        inputs = [b.to(self.device) for b in batch]
        if len(inputs) == 2:
            x, y = inputs
            t_obs = torch.full((x.size(0),), -1, dtype=torch.long, device=self.device)
            inputs = [x, y, t_obs]
        if hasattr(self.model, "loss"):
            return self.model.loss(*inputs)
        # fall back to assuming the model itself returns a loss
        output = self.model(*inputs)
        if isinstance(output, torch.Tensor):
            return output
        raise ValueError("Model must implement a 'loss' method or return a loss tensor")

    def fit(self, num_epochs: int) -> None:
        """Train the model for a fixed number of epochs."""
        for epoch in range(num_epochs):
            self.model.train()
            num_batches = len(self.train_loader)
            if self.logger:
                self.logger.start_epoch(epoch + 1, num_batches)
            for batch_idx, batch in enumerate(self.train_loader):
                X, Y, T_obs = self._extract_batch(batch)
                out = self.step(batch)
                loss = out["loss"] if isinstance(out, Mapping) else out
                self.optimizer.zero_grad()
                loss.backward()
                self._clip_grads()
                self.optimizer.step()
                if self.logger:
                    metrics = dict(self._metrics_from_loss(out))
                    metrics.update(self._treatment_metrics(X, Y, T_obs))
                    metrics.update(self._outcome_metrics(X, Y, T_obs))
                    self.logger.log_step(epoch + 1, batch_idx, num_batches, metrics)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.logger and self.val_loader is not None:
                val_metrics = self._eval_metrics(self.val_loader)
                self.logger.log_validation(epoch + 1, val_metrics)
            if self.logger:
                self.logger.end_epoch(epoch + 1)

    def evaluate(self, data_loader: Iterable) -> float:
        metrics = self._eval_metrics(data_loader)
        return metrics.get("loss", next(iter(metrics.values()), 0.0))

    def predict(self, *inputs: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            inputs = [i.to(self.device) for i in inputs]
            if hasattr(self.model, "predict"):
                return self.model.predict(*inputs)
            return self.model(*inputs)


__all__ = ["SupervisedTrainer"]
