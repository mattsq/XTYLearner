"""Probabilistic Circuit model using SPFlow.

This class wraps the reference implementation from the README for
learning a Sum-Product Network over tabular data with potentially
missing treatment labels.  It exposes minimal ``fit`` and
``predict_t_posterior`` methods.
"""

from __future__ import annotations

from typing import Iterable, Sequence, List

import math
import numpy as np
import pandas as pd

try:  # newer package name
    from spflow.inference.Inference import log_likelihood
    from spflow.learning.parametric.MSPN import learn_parametric
    from spflow.structure.Base import get_nodes_by_type
    from spflow.structure.Model import Product, Sum
    from spflow.structure.leaves.parametric.Parametric import (
        Categorical,
        Gaussian,
    )

    _HAS_SPFLOW = True
except ModuleNotFoundError:  # fallback to legacy import or no spflow
    try:
        from spn.algorithms.Inference import log_likelihood
        from spn.algorithms.LearningWrappers import learn_parametric
        from spn.structure.Base import get_nodes_by_type, Product, Sum
        from spn.structure.leaves.parametric.Parametric import (
            Categorical,
            Gaussian,
        )

        _HAS_SPFLOW = True
    except ModuleNotFoundError:  # completely unavailable
        _HAS_SPFLOW = False

from sklearn.linear_model import LogisticRegression, LinearRegression

from .registry import register_model


@register_model("prob_circuit")
class ProbCircuitModel:
    """Learn a probabilistic circuit over ``(X, T, Y)`` using SPFlow."""

    def __init__(
        self,
        min_instances_slice: int = 200,
        *,
        d_x: int | None = None,
        d_y: int | None = None,
        k: int | None = None,
    ) -> None:
        self.min_instances_slice = min_instances_slice
        self.root = None
        self._lr = None
        self._cols_num: Sequence[str] = []
        self._y_cols: List[str] = []
        self._t_idx: int = -1
        self._regs: list[LinearRegression] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        df_or_x: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        t: np.ndarray | None = None,
    ) -> "ProbCircuitModel":
        """Learn the circuit structure and parameters.

        The model can be trained directly from a :class:`pandas.DataFrame`
        containing covariates ``X*`` together with a treatment column ``T``
        and an outcome column ``Y``.  Alternatively, ``(X, Y, T)`` arrays can
        be provided.
        Treatment values equal to ``-1`` are interpreted as missing and will be
        treated as latent during learning.
        """

        global _HAS_SPFLOW
        if isinstance(df_or_x, pd.DataFrame):
            df = df_or_x.copy()
        else:
            if y is None or t is None:
                raise TypeError(
                    "y and t must be provided when fitting from array inputs"
                )
            X = df_or_x
            df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
            y_np = np.asarray(y)
            if y_np.ndim == 1 or (y_np.ndim == 2 and y_np.shape[1] == 1):
                df["Y"] = y_np.reshape(-1)
                self._y_cols = ["Y"]
            else:
                y_np = y_np.reshape(y_np.shape[0], -1)
                self._y_cols = [f"Y{i}" for i in range(y_np.shape[1])]
                for i, col in enumerate(self._y_cols):
                    df[col] = y_np[:, i]
            df["T"] = t
            df.loc[df["T"] == -1, "T"] = np.nan

        cols = list(df.columns)
        if "T" not in cols:
            raise ValueError("DataFrame must contain column 'T'")
        if not self._y_cols:
            if "Y" in cols:
                self._y_cols = ["Y"]
            else:
                self._y_cols = [c for c in cols if c.startswith("Y")]
                if not self._y_cols:
                    raise ValueError("DataFrame must contain outcome column(s) 'Y'")

        self._cols_num = [c for c in cols if c not in set(["T"] + self._y_cols)]
        self._t_idx = len(self._cols_num)

        data_np = df[self._cols_num + ["T"] + self._y_cols].to_numpy(dtype=np.float64)

        if _HAS_SPFLOW:
            meta_types = [Gaussian] * len(self._cols_num)
            meta_types.append(Categorical)  # T
            meta_types.extend([Gaussian] * len(self._y_cols))  # Y columns

            try:
                try:
                    self.root = learn_parametric(
                        data_np,
                        distributions=meta_types,
                        columns_to_learn=list(range(data_np.shape[1])),
                        min_instances_slice=self.min_instances_slice,
                    )
                except TypeError:
                    from spn.structure.Base import Context

                    context = Context(parametric_types=meta_types).add_domains(data_np)
                    self.root = learn_parametric(
                        data_np,
                        context,
                        min_instances_slice=self.min_instances_slice,
                    )

                n_sums = len(get_nodes_by_type(self.root, Sum))
                n_prods = len(get_nodes_by_type(self.root, Product))
                print(f"learned PC with {n_sums + n_prods} internal nodes")
            except Exception:  # fallback if SPFlow fails at runtime
                _HAS_SPFLOW = False
                labelled = df.dropna(subset=["T"])
                Xy = labelled[self._cols_num + self._y_cols].to_numpy(dtype=np.float64)
                t = labelled["T"].to_numpy().astype(int)
                self._lr = LogisticRegression().fit(Xy, t)
        else:
            labelled = df.dropna(subset=["T"])
            Xy = labelled[self._cols_num + self._y_cols].to_numpy(dtype=np.float64)
            t = labelled["T"].to_numpy().astype(int)
            self._lr = LogisticRegression().fit(Xy, t)

        # Fit simple regressors for E[Y|X,T] from labelled data
        self._regs.clear()
        if len(self._y_cols) > 0:
            labelled = df.dropna(subset=["T"])
            X_arr = labelled[self._cols_num].to_numpy(dtype=np.float64)
            Y_arr = labelled[self._y_cols].to_numpy(dtype=np.float64)
            t_arr = labelled["T"].to_numpy().astype(int)
            for val in sorted(np.unique(t_arr)):
                mask = t_arr == val
                reg = LinearRegression()
                if mask.sum() > 0:
                    reg.fit(X_arr[mask], Y_arr[mask])
                else:
                    reg.fit(np.zeros((1, X_arr.shape[1])), np.zeros((1, Y_arr.shape[1])))
                self._regs.append(reg)
        return self

    # ------------------------------------------------------------------
    def _posterior_row(self, row: Iterable[float]) -> float:
        if _HAS_SPFLOW:
            assert self.root is not None
            row0 = np.array(row, copy=True)
            row0[self._t_idx] = 0.0
            row1 = np.array(row, copy=True)
            row1[self._t_idx] = 1.0
            ll0 = log_likelihood(self.root, row0.reshape(1, -1))[0, 0]
            ll1 = log_likelihood(self.root, row1.reshape(1, -1))[0, 0]
            p1 = math.exp(ll1) / (math.exp(ll0) + math.exp(ll1))
            return p1
        assert self._lr is not None
        arr = np.asarray(row, dtype=float)
        feats = np.concatenate([arr[: self._t_idx], arr[self._t_idx + 1 :]])
        prob = self._lr.predict_proba(feats.reshape(1, -1))[0, 1]
        return float(prob)

    def predict_t_posterior(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return ``p(T=1 | X, Y)`` for each row.

        Parameters
        ----------
        X:
            Array of covariates with shape ``(n, d)``.
        Y:
            Outcome values ``(n,)`` or ``(n, 1)``.
        """

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.ndim != 2:
            raise ValueError("Y must be one- or two-dimensional")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")

        out = np.empty(X.shape[0])
        if _HAS_SPFLOW:
            for i in range(X.shape[0]):
                row = np.concatenate(
                    [X[i].astype(float), [np.nan], Y[i].astype(float)]
                )
                out[i] = self._posterior_row(row)
        else:
            data = np.concatenate([X, Y], axis=1)
            out[:] = self._lr.predict_proba(data)[:, 1]
        return out

    # --------------------------------------------------------------
    def predict_treatment_proba(
        self, X_or_Z: np.ndarray, Y: np.ndarray | None = None
    ) -> np.ndarray:
        """Return ``p(t|x,y)`` as a two-column array.

        Parameters
        ----------
        X_or_Z:
            Either the concatenated ``[X, Y]`` array of shape ``(n, d_x+d_y)``
            or just the covariate matrix ``X`` when ``Y`` is provided
            separately.
        Y:
            Optional outcome array of shape ``(n, d_y)``.  When ``None``,
            ``X_or_Z`` is assumed to contain both ``X`` and ``Y``.
        """

        if Y is None:
            Z = np.asarray(X_or_Z, dtype=float)
            X = Z[:, : self._t_idx]
            Y = Z[:, self._t_idx:]
        else:
            X = np.asarray(X_or_Z, dtype=float)
            Y = np.asarray(Y, dtype=float)

        p1 = self.predict_t_posterior(X, Y)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)

    # --------------------------------------------------------------
    def predict_outcome(self, X: np.ndarray, t: int | np.ndarray) -> np.ndarray:
        """Return expected outcome ``E[Y|X,T=t]`` using fitted regressors."""

        if not self._regs:
            raise ValueError("Model is not fitted")

        t_arr = np.asarray(t)
        if t_arr.ndim == 0:
            preds = self._regs[int(t_arr)].predict(X)
        else:
            preds = np.empty((len(t_arr), len(self._y_cols)))
            for val in np.unique(t_arr):
                mask = t_arr == val
                preds[mask] = self._regs[int(val)].predict(X[mask])

        if preds.shape[1] == 1:
            return preds.reshape(-1)
        return preds


__all__ = ["ProbCircuitModel"]
