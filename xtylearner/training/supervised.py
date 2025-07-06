from __future__ import annotations

from typing import Iterable

import torch

from .base_trainer import BaseTrainer


class SupervisedTrainer(BaseTrainer):
    """Generic trainer for fully observed models with a ``loss`` method."""

    def step(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
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
        for epoch in range(num_epochs):
            self.model.train()
            num_batches = len(self.train_loader)
            if self.logger:
                self.logger.start_epoch(epoch + 1, num_batches)
            for batch_idx, batch in enumerate(self.train_loader):
                X, Y, T_obs = self._extract_batch(batch)
                loss = self.step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.logger:
                    metrics = dict(self._metrics_from_loss(loss))
                    metrics.update(self._treatment_metrics(X, Y, T_obs))
                    self.logger.log_step(epoch + 1, batch_idx, num_batches, metrics)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.logger:
                self.logger.end_epoch(epoch + 1)

    def evaluate(self, data_loader: Iterable) -> float:
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in data_loader:
                loss = self.step(batch)
                total += float(loss.item()) * len(batch[0])
                n += len(batch[0])
        return total / max(n, 1)

    def predict(self, *inputs: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            inputs = [i.to(self.device) for i in inputs]
            if hasattr(self.model, "predict"):
                return self.model.predict(*inputs)
            return self.model(*inputs)


__all__ = ["SupervisedTrainer"]
