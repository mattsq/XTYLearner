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
            if self.logger and self.val_loader is not None:
                val_metrics = self._eval_metrics(self.val_loader)
                self.logger.log_validation(epoch + 1, val_metrics)
            if self.logger:
                self.logger.end_epoch(epoch + 1)

    def evaluate(self, data_loader: Iterable) -> float:
        metrics = self._eval_metrics(data_loader)
        return metrics.get("loss", next(iter(metrics.values()), 0.0))

    def predict(self, *args):
        """Return model predictions or samples.

        ``predict(n)`` draws ``n`` samples from the model prior using
        ``model.sample`` when called with a single integer argument.  When
        supplied a covariate matrix ``x`` and treatment indicator ``t``, the
        method returns outcome predictions by delegating to either
        ``model.predict_outcome`` or ``model.paired_sample``.
        """

        self.model.eval()
        with torch.no_grad():
            # Sampling from the prior
            if len(args) == 1 and isinstance(args[0], int):
                return self.model.sample(args[0])

            if len(args) != 2:
                raise TypeError("predict() expects `n` or `(x, t)` arguments")

            x, t = args
            x = x.to(self.device)

            if isinstance(t, int):
                t_tensor = torch.full((x.size(0),), t, dtype=torch.long, device=self.device)
            else:
                t_tensor = t.to(self.device)

            if hasattr(self.model, "predict_outcome"):
                return self.model.predict_outcome(x, t_tensor)

            if hasattr(self.model, "paired_sample"):
                y_all = self.model.paired_sample(x)
                if isinstance(t, torch.Tensor) and t.dim() > 0:
                    preds = torch.zeros_like(y_all[0])
                    for val in t_tensor.unique():
                        idx = t_tensor == val
                        preds[idx] = y_all[int(val)][idx]
                    return preds
                return y_all[int(t_tensor[0].item())]

            raise ValueError("Model does not support outcome prediction")


__all__ = ["DiffusionTrainer"]
