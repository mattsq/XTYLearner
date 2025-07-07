from __future__ import annotations

from typing import Iterable

import torch

from .base_trainer import BaseTrainer


class DiffusionTrainer(BaseTrainer):
    """Trainer for score-based diffusion models like JSBF."""

    def step(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        x, y, t = [b.to(self.device) for b in batch]
        return self.model.loss(x, y, t)

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
                self._clip_grads()
                self.optimizer.step()
                if self.logger:
                    metrics = dict(self._metrics_from_loss(loss))
                    metrics.update(self._treatment_metrics(X, Y, T_obs))
                    metrics.update(self._outcome_metrics(X, Y, T_obs))
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

    def predict(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            return self.model.sample(n)


__all__ = ["DiffusionTrainer"]
