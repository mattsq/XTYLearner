"""Lightweight wrapper around scikit-learn's :class:`LabelPropagation`."""

from __future__ import annotations

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
    """

    # ``target`` tells :class:`~ArrayTrainer` to use the treatment column.
    target = "treatment"
    requires_outcome = False

    def __init__(self, n_neighbors: int = 10):
        self.clf = LabelPropagation(kernel="knn", n_neighbors=n_neighbors)

    # ------------------------------------------------------------------
    def fit(self, X, y, t_obs=None):
        target = t_obs if t_obs is not None else y
        self.clf.fit(X, target)
        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        return self.clf.predict(X)

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    # ``ArrayTrainer`` looks for this method when computing metrics.
    predict_treatment_proba = predict_proba
