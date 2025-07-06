"""Probabilistic Circuit model using SPFlow.

This class wraps the reference implementation from the README for
learning a Sum-Product Network over tabular data with potentially
missing treatment labels.  It exposes minimal ``fit`` and
``predict_t_posterior`` methods.
"""

from __future__ import annotations

from typing import Iterable, Sequence

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

from sklearn.linear_model import LogisticRegression

from .registry import register_model


@register_model("prob_circuit")
class ProbCircuitModel:
    """Learn a probabilistic circuit over ``(X, T, Y)`` using SPFlow."""

    def __init__(self, min_instances_slice: int = 200) -> None:
        self.min_instances_slice = min_instances_slice
        self.root = None
        self._lr = None
        self._cols_num: Sequence[str] = []
        self._t_idx: int = -1

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "ProbCircuitModel":
        """Learn the circuit structure and parameters from ``df``.

        Parameters
        ----------
        df:
            DataFrame containing covariates ``X*``, a treatment column ``T``
            and an outcome column ``Y``.  Treatment may contain ``NaN`` which
            will be treated as latent during learning.
        """

        global _HAS_SPFLOW
        cols = list(df.columns)
        if "T" not in cols or "Y" not in cols:
            raise ValueError("DataFrame must contain columns 'T' and 'Y'")

        self._cols_num = [c for c in cols if c not in {"T", "Y"}]
        self._t_idx = len(self._cols_num)

        data_np = df[self._cols_num + ["T", "Y"]].to_numpy(dtype=np.float64)

        if _HAS_SPFLOW:
            meta_types = [Gaussian] * len(self._cols_num)
            meta_types.append(Categorical)  # T
            meta_types.append(Gaussian)  # Y

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
                Xy = labelled[self._cols_num + ["Y"]].to_numpy(dtype=np.float64)
                t = labelled["T"].to_numpy().astype(int)
                self._lr = LogisticRegression().fit(Xy, t)
        else:
            labelled = df.dropna(subset=["T"])
            Xy = labelled[self._cols_num + ["Y"]].to_numpy(dtype=np.float64)
            t = labelled["T"].to_numpy().astype(int)
            self._lr = LogisticRegression().fit(Xy, t)
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
        prob = self._lr.predict_proba(np.array(row[:-1]).reshape(1, -1))[0, 1]
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

        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y[:, 0]
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")

        out = np.empty(X.shape[0])
        if _HAS_SPFLOW:
            for i in range(X.shape[0]):
                row = np.concatenate([X[i].astype(float), [np.nan], [float(Y[i])]])
                out[i] = self._posterior_row(row)
        else:
            data = np.concatenate([X, Y.reshape(-1, 1)], axis=1)
            out[:] = self._lr.predict_proba(data)[:, 1]
        return out

    # --------------------------------------------------------------
    def predict_treatment_proba(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return ``p(t|x,y)`` as a two-column array."""

        p1 = self.predict_t_posterior(X, Y)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


__all__ = ["ProbCircuitModel"]
