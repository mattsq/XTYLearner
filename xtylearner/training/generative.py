from __future__ import annotations

from typing import Iterable

import torch
from torch.nn.functional import one_hot

from .base_trainer import BaseTrainer
from ..models.generative import M2VAE, SS_CEVAE


class GenerativeTrainer(BaseTrainer):
    """Trainer for generative models using an ELBO objective."""

    def step(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        data = [b.to(self.device) for b in batch]
        if len(data) == 2:
            x, y = data
            t = torch.full((x.size(0),), -1, dtype=torch.long, device=self.device)
        else:
            x, y, t = data
        return self.model.elbo(x, y, t)

    def fit(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            self.model.train()
            num_batches = len(self.train_loader)
            if self.logger:
                self.logger.start_epoch(epoch + 1, num_batches)
            for batch_idx, batch in enumerate(self.train_loader):
                loss = self.step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.logger:
                    metrics = self._metrics_from_loss(loss)
                    self.logger.log_step(epoch + 1, batch_idx, num_batches, metrics)
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

    def predict(self, x: torch.Tensor, t_val: int) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            z_dim = self.model.enc_z.net_mu[-1].out_features
            z = torch.randn(x.size(0), z_dim, device=self.device)
            t1h = one_hot(
                torch.full((x.size(0),), t_val, device=self.device), self.model.k
            ).float()
            if isinstance(self.model, M2VAE):
                return self.model.dec_y(x.to(self.device), t1h, z)
            elif isinstance(self.model, SS_CEVAE):
                return self.model.dec_y(z, x.to(self.device), t1h)
            else:
                raise ValueError("Unsupported model type for prediction")


__all__ = ["GenerativeTrainer"]
