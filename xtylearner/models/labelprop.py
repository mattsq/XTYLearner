from sklearn.semi_supervised import LabelPropagation

from .registry import register_model


@register_model("lp_knn")
class LP_KNN:
    """k-NN label propagation baseline."""

    def __init__(self, n_neighbors: int = 10):
        self.clf = LabelPropagation(kernel="knn", n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
