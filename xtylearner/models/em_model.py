# em_baseline.py
# Dempster, Laird & Rubin “Maximum Likelihood from Incomplete Data via the EM Algorithm”, JRSS-B 1977
# link.springer.com
# Nigam et al. “Text Classification from Labeled and Unlabeled Documents using EM”, MLJ 2000
# link.springer.com

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import norm

from .registry import register_model


def em_learn(
    X,
    Y,
    T_obs,  # arrays of shape (n, d), (n,), (n,) -- T_obs may have -1 for "missing"
    k,
    max_iter=20,
    tol=1e-4,
    classifier_factory=None,
    regressor_factory=None,
    verbose=False,
):
    """Return:
    - clf_T      : classifier modelling  p(T | X)
    - regs_Y[t]  : list of regressors modelling E[Y | X, T=t]
    - sigma2[t]  : residual variance per treatment
    - T_imputed  : np.ndarray of shape (n,) with hard EM labels
    """
    # ----- split labelled vs unlabelled ----------------------------------
    labelled = T_obs != -1
    unlabelled = ~labelled
    T = T_obs.copy()

    # ----- helpers --------------------------------------------------------
    if classifier_factory is None:

        def classifier_factory() -> LogisticRegression:
            if k > 2:
                return LogisticRegression(
                    multi_class="multinomial", solver="lbfgs", max_iter=200
                )
            return LogisticRegression(max_iter=200)

    if regressor_factory is None:

        def regressor_factory() -> LinearRegression:
            return LinearRegression()

    def fit_classifier(X_, T_):
        clf = classifier_factory()
        clf.fit(X_, T_)
        return clf

    def fit_regressions(X_, Y_, T_):
        regs, sig2 = [], []
        for t in range(k):
            mask = T_ == t
            # If a treatment is absent in current pseudo-labels, keep a dummy regressor
            if mask.sum() < 2:
                dummy = regressor_factory()
                dummy.fit(np.zeros((1, X_.shape[1])), [Y_.mean()])
                regs.append(dummy)
                sig2.append(Y_.var() + 1e-6)
                continue
            r = regressor_factory().fit(X_[mask], Y_[mask])
            resid = Y_[mask] - r.predict(X_[mask])
            regs.append(r)
            sig2.append(resid.var() + 1e-6)  # avoid zero variance
        return regs, np.array(sig2)

    # ----- initialisation: supervise only on labelled data ---------------
    clf_T = fit_classifier(X[labelled], T[labelled])
    regs_Y, s2 = fit_regressions(X[labelled], Y[labelled], T[labelled])

    last_ll = -np.inf
    for it in range(max_iter):
        # ===== E-step: impute missing T via MAP ===========================
        if unlabelled.sum() > 0:
            log_post = np.zeros((unlabelled.sum(), k))
            p_t_given_x = clf_T.predict_proba(X[unlabelled])  # shape (n_U, K)
            y_u = Y[unlabelled]

            for t in range(k):
                mu = regs_Y[t].predict(X[unlabelled])
                log_lik = norm.logpdf(y_u, loc=mu, scale=np.sqrt(s2[t]))
                log_post[:, t] = np.log(p_t_given_x[:, t] + 1e-12) + log_lik

            T_hat = log_post.argmax(axis=1)
            T[unlabelled] = T_hat

        # ===== M-step: refit heads on (labelled + pseudo-labelled) =======
        clf_T = fit_classifier(X, T)
        regs_Y, s2 = fit_regressions(X, Y, T)

        # ===== Convergence check: complete-data log-likelihood ===========
        ll = 0.0
        # p(T|X) term
        if hasattr(clf_T, "predict_log_proba"):
            ll += clf_T.predict_log_proba(X)[np.arange(len(T)), T].sum()
        else:
            probs = clf_T.predict_proba(X)
            ll += np.log(probs[np.arange(len(T)), T] + 1e-12).sum()
        # p(Y|X,T) term
        for t in range(k):
            mask = T == t
            mu = regs_Y[t].predict(X[mask])
            ll += norm.logpdf(Y[mask], loc=mu, scale=np.sqrt(s2[t])).sum()

        if verbose:
            print(f"iter {it+1:2d}: complete-data LL = {ll:.2f}")
        if np.abs(ll - last_ll) < tol:
            break
        last_ll = ll

    return clf_T, regs_Y, s2, T


@register_model("em")
class EMModel:
    """Wrapper around :func:`em_learn` for the model registry."""

    def __init__(
        self,
        k: int,
        *,
        max_iter: int = 20,
        tol: float = 1e-4,
        classifier_factory=None,
        regressor_factory=None,
        verbose: bool = False,
    ) -> None:
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.classifier_factory = classifier_factory
        self.regressor_factory = regressor_factory
        self.verbose = verbose
        self.clf_T = None
        self.regs_Y = None
        self.sigma2 = None
        self.T_imputed = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray, T_obs: np.ndarray) -> "EMModel":
        """Fit the EM model on the provided data."""

        (
            self.clf_T,
            self.regs_Y,
            self.sigma2,
            self.T_imputed,
        ) = em_learn(
            X,
            Y,
            T_obs,
            self.k,
            max_iter=self.max_iter,
            tol=self.tol,
            classifier_factory=self.classifier_factory,
            regressor_factory=self.regressor_factory,
            verbose=self.verbose,
        )
        return self

    # ------------------------------------------------------------------
    def predict_treatment_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``p(T | X)`` using the learned classifier."""

        if self.clf_T is None:
            raise ValueError("Model is not fitted")
        return self.clf_T.predict_proba(X)

    # ------------------------------------------------------------------
    def predict_outcome(self, X: np.ndarray, t: int) -> np.ndarray:
        """Predict the outcome for treatment ``t``."""

        if self.regs_Y is None:
            raise ValueError("Model is not fitted")
        return self.regs_Y[t].predict(X)


__all__ = ["em_learn", "EMModel"]
