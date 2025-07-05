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
        for _ in range(num_epochs):
            self.model.train()
            for batch in self.train_loader:
                loss = self.step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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
