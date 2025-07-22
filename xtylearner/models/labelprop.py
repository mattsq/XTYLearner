"""Lightweight wrapper around scikit-learn's :class:`LabelPropagation`."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.semi_supervised import LabelPropagation

from .registry import register_model


@register_model("lp_knn")
class LP_KNN:
    """k-NN label propagation baseline.

    The model propagates *treatment* labels through an affinity graph.  It can
    be used with :class:`~xtylearner.training.ArrayTrainer`, which will pass the
    observed treatment column as the target argument.  For compatibility with the
    registry, the ``fit`` method also accepts datasets without an explicit
    treatment column and will fall back to the provided ``y`` labels.

    Parameters
    ----------
    n_neighbors:
        Number of neighbours for the underlying ``LabelPropagation`` classifier.
    regressor:
        Optional scikit-learn regressor.  When provided, ``fit`` trains this
        regressor on the full dataset using the propagated labels appended as
        one-hot features.  This enables :meth:`predict_outcome` and
        :meth:`regressor_metrics` for outcome prediction and evaluation.
    """

    # ``target`` tells :class:`~ArrayTrainer` to use the treatment column.
    target = "treatment"
    requires_outcome = False

    def __init__(
        self,
        n_neighbors: int = 10,
        regressor=None,
        *,
        d_x: int | None = None,
        d_y: int | None = None,
        k: int | None = None,
    ):
        self.clf = LabelPropagation(kernel="knn", n_neighbors=n_neighbors)
        self.regressor = regressor
        self.k = k

    # ------------------------------------------------------------------
    def fit(self, X, y, t_obs=None):
        target = t_obs if t_obs is not None else y
        self.clf.fit(X, target)
        self.k = len(self.clf.classes_)

        if self.regressor is not None:
            t_full = self.clf.transduction_
            X_reg = np.concatenate([X, self._one_hot(t_full)], axis=1)
            self.regressor.fit(X_reg, y)
        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        return self.clf.predict(X)

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    # ``ArrayTrainer`` looks for this method when computing metrics.
    predict_treatment_proba = predict_proba

    # ------------------------------------------------------------------
    def _one_hot(self, t):
        if self.k is None:
            raise ValueError("Model not fitted")
        t = np.asarray(t, dtype=int)
        return np.eye(self.k)[t]

    # ------------------------------------------------------------------
    def predict_outcome(self, X, t):
        if self.regressor is None:
            t_arr = np.asarray(t)
            if t_arr.ndim == 1:
                raise ValueError("A regressor must be provided")
            return np.zeros((len(X), 1))
        t_arr = np.asarray(t, dtype=int)
        X_reg = np.concatenate([X, self._one_hot(t_arr)], axis=1)
        return self.regressor.predict(X_reg)

    # ------------------------------------------------------------------
    def regressor_metrics(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        T_obs: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        T_val: np.ndarray,
    ) -> dict[str, float]:
        """Return metrics comparing labelled vs full-data regressors."""

        if self.regressor is None:
            return {}

        mask = T_obs != -1
        if mask.any():
            reg_lab = clone(self.regressor)
            reg_lab.fit(
                np.concatenate([X_train[mask], self._one_hot(T_obs[mask])], axis=1),
                Y_train[mask],
            )
            pred_lab = reg_lab.predict(
                np.concatenate([X_val, self._one_hot(T_val)], axis=1)
            )
            rmse_lab = float(np.sqrt(((pred_lab - Y_val) ** 2).mean()))
        else:
            rmse_lab = float("nan")

        pred_full = self.predict_outcome(X_val, T_val)
        rmse_full = float(np.sqrt(((pred_full - Y_val) ** 2).mean()))

        mask_val = T_val != -1
        if mask_val.any():
            acc = float((self.predict(X_val[mask_val]) == T_val[mask_val]).mean())
        else:
            acc = float("nan")

        return {
            "val_acc": acc,
            "rmse_labelled": rmse_lab,
            "rmse_full": rmse_full,
        }
