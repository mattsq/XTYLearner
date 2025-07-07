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
        self.model.fit(X, Y, T_obs)

    def evaluate(self, data_loader: Iterable) -> float:
        X, Y, T_obs = self._collect_arrays(data_loader)
        mask = T_obs != -1
        if mask.sum() == 0:
            return 0.0
        probs = self.model.predict_treatment_proba(X[mask])
        nll = -np.log(probs[np.arange(mask.sum()), T_obs[mask]] + 1e-12).mean()
        return float(nll)

    def predict(self, x: torch.Tensor, t_val: int):
        X_np = x.cpu().numpy()
        return self.model.predict_outcome(X_np, t_val)

    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        X_np = x.cpu().numpy()
        probs = self.model.predict_treatment_proba(X_np)
        return torch.from_numpy(probs)


__all__ = ["EMTrainer"]
