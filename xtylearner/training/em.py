from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .base_trainer import BaseTrainer


class EMTrainer(BaseTrainer):
    """Trainer for :class:`~xtylearner.models.em_model.EMModel`."""

    def _collect_arrays(
        self, loader: Iterable
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_list, Y_list, T_list = [], [], []
        for batch in loader:
            x, y, t = self._extract_batch(batch)
            X_list.append(x.cpu().numpy())
            Y_list.append(y.squeeze(-1).cpu().numpy())
            T_list.append(t.cpu().numpy())
        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)
        T = np.concatenate(T_list, axis=0)
        return X, Y, T

    def fit(self, num_epochs: int) -> None:
        X, Y, T_obs = self._collect_arrays(self.train_loader)
        num_batches = len(self.train_loader)
        if self.logger:
            self.logger.start_epoch(1, num_batches)
        self.model.fit(X, Y, T_obs)
        if self.logger:
            metrics = self._treatment_metrics(
                torch.from_numpy(X),
                torch.from_numpy(Y).unsqueeze(-1),
                torch.from_numpy(T_obs),
            )
            self.logger.log_step(1, num_batches - 1, num_batches, metrics)
            self.logger.end_epoch(1)

    def _treatment_metrics(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> dict[str, float]:
        if self.model.log_likelihood is None:
            return {}
        return {"cd_ll": float(self.model.log_likelihood)}

    def evaluate(self, data_loader: Iterable) -> float:
        X, Y, T_obs = self._collect_arrays(data_loader)
        mask = T_obs != -1
        if mask.sum() == 0:
            return 0.0
        Z = np.concatenate([X[mask], Y[mask, None]], axis=1)
        probs = self.model.predict_treatment_proba(Z)
        nll = -np.log(probs[np.arange(mask.sum()), T_obs[mask]] + 1e-12).mean()
        return float(nll)

    def predict(self, x: torch.Tensor, t_val: int):
        X_np = x.cpu().numpy()
        return self.model.predict_outcome(X_np, t_val)

    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        X_np = x.cpu().numpy()
        y_np = y.squeeze(-1).cpu().numpy()
        Z = np.concatenate([X_np, y_np[:, None]], axis=1)
        probs = self.model.predict_treatment_proba(Z)
        return torch.from_numpy(probs)


__all__ = ["EMTrainer"]
