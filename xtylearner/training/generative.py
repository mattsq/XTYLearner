from __future__ import annotations

from typing import Iterable

import torch
from torch.nn.functional import one_hot

from .base_trainer import BaseTrainer
from ..models.generative import M2VAE, SS_CEVAE, DiffusionCEVAE


class GenerativeTrainer(BaseTrainer):
    """Trainer for generative models using an ELBO objective."""

    def step(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        """Compute the ELBO (or loss) for one mini-batch.

        Parameters
        ----------
        batch:
            Iterable providing ``(X, Y, T_obs)`` tensors. When only ``X`` and
            ``Y`` are present a dummy ``T`` filled with ``-1`` is assumed.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor to backpropagate.
        """
        data = [b.to(self.device) for b in batch]
        if len(data) == 2:
            x, y = data
            t = torch.full((x.size(0),), -1, dtype=torch.long, device=self.device)
        else:
            x, y, t = data
        if hasattr(self.model, "loss"):
            return self.model.loss(x, y, t)
        return self.model.elbo(x, y, t)

    def fit(self, num_epochs: int) -> None:
        """Run the training loop for a fixed number of epochs.

        Parameters
        ----------
        num_epochs:
            Number of passes over ``train_loader`` to perform.
        """
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
        return {
            "loss": float(loss_val),
            "treatment accuracy": float(metrics.get("accuracy", 0.0)),
            "outcome rmse": float(metrics.get("rmse", 0.0)),
            "outcome rmse labelled": float(metrics.get("rmse_labelled", 0.0)),
            "outcome rmse unlabelled": float(metrics.get("rmse_unlabelled", 0.0)),
        }

    def predict(self, x: torch.Tensor, t_val: int) -> torch.Tensor:
        """Predict outcomes for a fixed treatment value.

        Parameters
        ----------
        x:
            Covariate matrix on the same device as the model.
        t_val:
            Integer treatment index to condition on.

        Returns
        -------
        torch.Tensor
            Predicted outcome tensor with one row per sample in ``x``.
        """

        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)

            # Use model-provided prediction when available
            if hasattr(self.model, "predict_outcome"):
                try:
                    return self.model.predict_outcome(x, t_val)
                except Exception:
                    pass
            if hasattr(self.model, "predict"):
                try:
                    return self.model.predict(x, t_val)
                except Exception:
                    pass
            try:
                return self.model(x, torch.full((x.size(0),), t_val, device=x.device))
            except Exception:
                pass

            # Fallback for VAE-style models
            if isinstance(self.model, DiffusionCEVAE):
                u = torch.randn(x.size(0), self.model.d_u, device=self.device)
                t = torch.full((x.size(0),), t_val, device=self.device)
                return self.model.dec_y(x, t, u)

            z_dim = self.model.enc_z.net_mu[-1].out_features
            z = torch.randn(x.size(0), z_dim, device=self.device)
            t1h = one_hot(
                torch.full((x.size(0),), t_val, device=self.device), self.model.k
            ).float()
            if isinstance(self.model, M2VAE):
                return self.model.dec_y(x, t1h, z)
            elif isinstance(self.model, SS_CEVAE):
                return self.model.dec_y(z, x, t1h)

            raise ValueError("Unsupported model type for prediction")


__all__ = ["GenerativeTrainer"]
