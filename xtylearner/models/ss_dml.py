import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from doubleml import DoubleMLData, DoubleMLSSM

from .registry import register_model


@register_model("ss_dml")
class SSDMLModel:
    """Semi-Supervised Double Machine Learning baseline."""

    k = 2

    def __init__(self, ml_g=None, ml_m=None, ml_pi=None, n_folds: int = 5, score: str = "missing-at-random"):
        self.ml_g = ml_g or RandomForestRegressor(min_samples_leaf=5)
        self.ml_m = ml_m or RandomForestClassifier()
        self.ml_pi = ml_pi or RandomForestClassifier()
        self.n_folds = n_folds
        self.score = score
        self._dml = None
        self.g_hat_ = None
        self.g_hat_0, self.g_hat_1 = None, None
        self.m_hat_ = None
        self.ate_ = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray, T_obs: np.ndarray):
        labelled = T_obs != -1
        X_L, Y_L, T_L = X[labelled], Y[labelled], T_obs[labelled]
        s = np.ones_like(T_L)
        data = DoubleMLData.from_arrays(x=X_L, y=Y_L, d=T_L.reshape(-1, 1), s=s)
        self._dml = DoubleMLSSM(
            data,
            ml_g=self.ml_g,
            ml_pi=self.ml_pi,
            ml_m=self.ml_m,
            n_folds=self.n_folds,
            score=self.score,
            normalize_ipw=True,
        )
        self._dml.fit()
        self.ate_ = float(self._dml.effect)
        self.g_hat_ = self._dml._g_hat
        self.g_hat_0, self.g_hat_1 = self.g_hat_
        self.m_hat_ = self._dml._m_hat
        return self

    # ------------------------------------------------------------------
    def predict_treatment_proba(self, Z: np.ndarray, *_):
        return self.m_hat_.predict_proba(Z)

    def predict_outcome(self, X: np.ndarray, t: int):
        return self.g_hat_[t].predict(X)

    @property
    def tau_(self):
        return self.ate_


__all__ = ["SSDMLModel"]
