from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .base_trainer import BaseTrainer


class ArrayTrainer(BaseTrainer):
    """Trainer for array-based models with a ``fit`` method.

    This generalises the previous :class:`EMTrainer` used for the EM baseline
    so that any model exposing a ``fit`` method operating on NumPy arrays can be
    trained.  It works with datasets providing ``(X, Y)`` pairs as well as
    ``(X, Y, T)`` triples – the latter will be passed to the model only if its
    ``fit`` signature accepts a treatment argument.
    """

    def _collect_arrays(
        self, loader: Iterable
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gather ``X``, ``Y`` and ``T`` arrays from ``loader``.

        Parameters
        ----------
        loader:
            Iterable yielding batches compatible with :meth:`_extract_batch`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Arrays ``(X, Y, T)`` concatenated over the loader.
        """
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
        """Fit the wrapped model using all available data.

        Parameters
        ----------
        num_epochs:
            Unused but kept for API compatibility.
        """
        X, Y, T_obs = self._collect_arrays(self.train_loader)
        num_batches = len(self.train_loader)
        if self.logger:
            self.logger.start_epoch(1, num_batches)
        import inspect

        fit_sig = inspect.signature(self.model.fit)
        params = [p.name for p in fit_sig.parameters.values() if p.name != "self"]
        use_t = getattr(self.model, "target", "outcome") == "treatment"
        if len(params) >= 3:
            self.model.fit(X, Y, T_obs)
        else:
            target = T_obs if use_t else Y
            self.model.fit(X, target)
        if self.logger:
            metrics = self._treatment_metrics(
                torch.from_numpy(X),
                torch.from_numpy(Y).unsqueeze(-1),
                torch.from_numpy(T_obs),
            )
            metrics.update(
                self._outcome_metrics(
                    torch.from_numpy(X),
                    torch.from_numpy(Y).unsqueeze(-1),
                    torch.from_numpy(T_obs),
                )
            )
            self.logger.log_step(1, num_batches - 1, num_batches, metrics)
            if self.val_loader is not None:
                Xv, Yv, Tv = self._collect_arrays(self.val_loader)
                val_metrics = self._treatment_metrics(
                    torch.from_numpy(Xv),
                    torch.from_numpy(Yv).unsqueeze(-1),
                    torch.from_numpy(Tv),
                )
                val_metrics.update(
                    self._outcome_metrics(
                        torch.from_numpy(Xv),
                        torch.from_numpy(Yv).unsqueeze(-1),
                        torch.from_numpy(Tv),
                    )
                )
                if hasattr(self.model, "regressor_metrics"):
                    extra = self.model.regressor_metrics(X, Y, T_obs, Xv, Yv, Tv)
                    val_metrics.update(extra)
                self.logger.log_validation(1, val_metrics)
            self.logger.end_epoch(1)

    def _treatment_metrics(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> dict[str, float]:
        # The EM baseline exposes a ``log_likelihood`` attribute after calling
        # ``fit``.  Older or simpler models such as :class:`LP_KNN` do not
        # implement this attribute.  In that case we fall back to the default
        # metric computation implemented in :class:`BaseTrainer`, which relies on
        # ``predict_treatment_proba`` when available.
        if hasattr(self.model, "log_likelihood"):
            if self.model.log_likelihood is None:
                return {}
            return {"cd_ll": float(self.model.log_likelihood)}

        return super()._treatment_metrics(x, y, t_obs)

    def evaluate(self, data_loader: Iterable) -> Mapping[str, float]:
        """Return evaluation metrics computed over ``data_loader``.

        Parameters
        ----------
        data_loader:
            Iterable of evaluation batches.

        Returns
        -------
        Mapping[str, float]
            Dictionary with loss, treatment accuracy and RMSE metrics.
        """
        X, Y, T_obs = self._collect_arrays(data_loader)
        metrics = {}
        if hasattr(self.model, "predict_treatment_proba"):
            mask = T_obs != -1
            if mask.sum() == 0:
                loss = 0.0
            else:
                if getattr(self.model, "requires_outcome", True):
                    Z = np.concatenate([X[mask], Y[mask, None]], axis=1)
                else:
                    Z = X[mask]
                probs = self.model.predict_treatment_proba(Z)
                loss = -np.log(probs[np.arange(mask.sum()), T_obs[mask]] + 1e-12).mean()
            metrics["loss"] = float(loss)
        else:
            preds = self.model.predict(X)
            target = T_obs if getattr(self.model, "target", "outcome") == "treatment" else Y
            acc = float((preds == target).mean())
            metrics["loss"] = acc
        tensor_X = torch.from_numpy(X)
        tensor_Y = torch.from_numpy(Y).unsqueeze(-1)
        tensor_T = torch.from_numpy(T_obs)
        metrics.update(self._treatment_metrics(tensor_X, tensor_Y, tensor_T))
        metrics.update(self._outcome_metrics(tensor_X, tensor_Y, tensor_T))
        return {
            "loss": float(metrics.get("loss", 0.0)),
            "treatment accuracy": float(metrics.get("accuracy", 0.0)),
            "outcome rmse": float(metrics.get("rmse", 0.0)),
        }

    def predict(self, x: torch.Tensor, t_val: int | None = None):
        """Return predictions for ``x`` using the underlying model.

        Parameters
        ----------
        x:
            Covariate matrix.
        t_val:
            Optional treatment value when calling outcome predictors.

        Returns
        -------
        Any
            ``numpy`` array of predictions.
        """
        X_np = x.cpu().numpy()
        if t_val is not None and hasattr(self.model, "predict_outcome"):
            return self.model.predict_outcome(X_np, t_val)
        return self.model.predict(X_np)

    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute ``p(t|x,y)`` for array-based models.

        Parameters
        ----------
        x:
            Covariate tensor.
        y:
            Outcome tensor.

        Returns
        -------
        torch.Tensor
            Probability matrix of shape ``(n, k)``.
        """
        X_np = x.cpu().numpy()
        if getattr(self.model, "requires_outcome", True):
            y_np = y.squeeze(-1).cpu().numpy()
            Z = np.concatenate([X_np, y_np[:, None]], axis=1)
        else:
            Z = X_np
        probs = self.model.predict_treatment_proba(Z)
        return torch.from_numpy(probs)


EMTrainer = ArrayTrainer

__all__ = ["ArrayTrainer", "EMTrainer"]
