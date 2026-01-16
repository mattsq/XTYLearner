from __future__ import annotations

from typing import Iterable, Mapping

import torch
import optuna

from .base_trainer import BaseTrainer


class SupervisedTrainer(BaseTrainer):
    """Generic trainer for fully observed models with a ``loss`` method."""

    def step(self, batch: Iterable[torch.Tensor]):
        """Forward ``batch`` through the model and return the loss.

        Parameters
        ----------
        batch:
            Iterable of tensors containing covariates ``X``, outcomes ``Y`` and
            optionally observed treatments ``T``.

        Returns
        -------
        torch.Tensor
            Loss tensor produced by the model.
        """
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
        """Optimise the model parameters for ``num_epochs`` epochs.

        Parameters
        ----------
        num_epochs:
            Number of passes over ``train_loader``.
        """
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
                # Update model state after optimizer step (e.g., EMA for mean teacher)
                if hasattr(self.model, "step") and callable(self.model.step):
                    self.model.step()
                if self.logger:
                    metrics = dict(self._metrics_from_loss(out))
                    metrics.update(self._treatment_metrics(X, Y, T_obs))
                    metrics.update(self._outcome_metrics(X, Y, T_obs))
                    self.logger.log_step(epoch + 1, batch_idx, num_batches, metrics)
            if self.scheduler is not None:
                self.scheduler.step()

            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self._eval_metrics(self.val_loader)
                if self.logger:
                    self.logger.log_validation(epoch + 1, val_metrics)

            if self.logger:
                self.logger.end_epoch(epoch + 1)

            if self.optuna_trial is not None and val_metrics is not None:
                metric = val_metrics.get("loss", next(iter(val_metrics.values()), 0.0))
                self.optuna_trial.report(metric, step=epoch + 1)
                if self.optuna_trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    def evaluate(self, data_loader: Iterable) -> Mapping[str, float]:
        """Return averaged metrics on ``data_loader``.

        Parameters
        ----------
        data_loader:
            Iterable yielding evaluation batches.

        Returns
        -------
        Mapping[str, float]
            Dictionary with loss, treatment accuracy and RMSE metrics.
        """
        metrics = self._eval_metrics(data_loader)
        loss_val = metrics.get("loss", next(iter(metrics.values()), 0.0))
        result = {
            "loss": float(loss_val),
            "treatment accuracy": float(metrics.get("accuracy", 0.0)),
            "outcome rmse": float(metrics.get("rmse", 0.0)),
            "outcome rmse labelled": float(metrics.get("rmse_labelled", 0.0)),
            "outcome rmse unlabelled": float(metrics.get("rmse_unlabelled", 0.0)),
        }
        # Add treatment rmse for continuous treatment models
        if "treatment_rmse" in metrics:
            result["treatment rmse"] = float(metrics["treatment_rmse"])
        return result

    def predict(self, *inputs: torch.Tensor):
        """Return model outputs for ``inputs``.

        Parameters
        ----------
        *inputs:
            Tensors forwarded to the underlying model.

        Returns
        -------
        torch.Tensor | Any
            Predictions returned by the model.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = [
                i.to(self.device) if isinstance(i, torch.Tensor) else i for i in inputs
            ]
            if hasattr(self.model, "predict"):
                return self.model.predict(*inputs)
            return self.model(*inputs)


__all__ = ["SupervisedTrainer"]
