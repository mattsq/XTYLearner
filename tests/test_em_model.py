import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor

from xtylearner.models.em_model import em_learn


def _generate_data(n=20, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    T_true = rng.integers(0, k, size=n)
    Y = T_true + rng.normal(size=n)
    mask = rng.random(n) < 0.3
    T_obs = T_true.copy()
    T_obs[mask] = -1
    return X, Y, T_obs


def test_em_learn_default_runs():
    X, Y, T = _generate_data(n=15, k=2, seed=1)
    clf, regs, s2, T_hat, ll = em_learn(X, Y, T, k=2, max_iter=2)
    assert len(regs) == 2
    assert T_hat.shape == (15,)
    assert hasattr(clf, "predict_proba")
    assert isinstance(ll, float)


def test_em_learn_custom_models():
    X, Y, T = _generate_data(n=10, k=3, seed=2)

    def clf_factory():
        return GaussianNB()

    def reg_factory():
        return DecisionTreeRegressor(random_state=0)

    clf, regs, s2, T_hat, ll = em_learn(
        X,
        Y,
        T,
        k=3,
        classifier_factory=clf_factory,
        regressor_factory=reg_factory,
        max_iter=2,
    )
    assert len(regs) == 3
    assert T_hat.shape == (10,)
    assert hasattr(clf, "predict_proba")
    assert isinstance(ll, float)
