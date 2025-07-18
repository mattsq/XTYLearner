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
    """Expectation-maximisation training loop.

    Parameters
    ----------
    X, Y, T_obs:
        Arrays containing covariates, outcomes and (possibly missing)
        treatments.
    k:
        Number of treatment categories.
    max_iter:
        Maximum number of EM iterations.
    tol:
        Stopping tolerance on the complete-data log-likelihood.
    classifier_factory:
        Callable returning a scikit-learn classifier for ``p(T|X,Y)``.
    regressor_factory:
        Callable returning regressors for ``E[Y|X,T]``.
    verbose:
        If ``True`` print log-likelihood progress.

    Returns
    -------
    clf_T : ``sklearn.base.ClassifierMixin``
        Fitted classifier over ``(X,Y)``.
    regs_Y : list
        List of regressors predicting ``Y`` given ``X`` and ``T``.
    sigma2 : np.ndarray
        Residual variance per treatment.
    T_imputed : np.ndarray
        Hard EM labels for ``T``.
    ll : float
        Final complete-data log-likelihood.
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

    def fit_classifier(X_, Y_, T_):
        Y_ = Y_.reshape(len(Y_), -1)
        Z = np.concatenate([X_, Y_], axis=1)
        clf = classifier_factory()
        clf.fit(Z, T_)
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
            sig2.append(resid.var(axis=0) + 1e-6)  # avoid zero variance
        return regs, np.array(sig2)

    # ----- initialisation: supervise only on labelled data ---------------
    clf_T = fit_classifier(X[labelled], Y[labelled], T[labelled])
    regs_Y, s2 = fit_regressions(X[labelled], Y[labelled], T[labelled])

    last_ll = -np.inf
    for it in range(max_iter):
        # ===== E-step: impute missing T via MAP ===========================
        if unlabelled.sum() > 0:
            log_post = np.zeros((unlabelled.sum(), k))
            Z_u = np.concatenate(
                [X[unlabelled], Y[unlabelled].reshape(unlabelled.sum(), -1)],
                axis=1,
            )
            p_t_given_x = clf_T.predict_proba(Z_u)  # shape (n_U, K)
            y_u = Y[unlabelled]

            for t in range(k):
                mu = regs_Y[t].predict(X[unlabelled])
                log_lik = norm.logpdf(y_u, loc=mu, scale=np.sqrt(s2[t]))
                if log_lik.ndim > 1:
                    log_lik = log_lik.sum(axis=1)
                log_post[:, t] = np.log(p_t_given_x[:, t] + 1e-12) + log_lik

            T_hat = log_post.argmax(axis=1)
            T[unlabelled] = T_hat

        # ===== M-step: refit heads on (labelled + pseudo-labelled) =======
        clf_T = fit_classifier(X, Y, T)
        regs_Y, s2 = fit_regressions(X, Y, T)

        # ===== Convergence check: complete-data log-likelihood ===========
        ll = 0.0
        # p(T|X) term
        Z_all = np.concatenate([X, Y.reshape(len(Y), -1)], axis=1)
        if hasattr(clf_T, "predict_log_proba"):
            ll += clf_T.predict_log_proba(Z_all)[np.arange(len(T)), T].sum()
        else:
            probs = clf_T.predict_proba(Z_all)
            ll += np.log(probs[np.arange(len(T)), T] + 1e-12).sum()
        # p(Y|X,T) term
        for t in range(k):
            mask = T == t
            mu = regs_Y[t].predict(X[mask])
            log_lik = norm.logpdf(Y[mask], loc=mu, scale=np.sqrt(s2[t]))
            if log_lik.ndim > 1:
                log_lik = log_lik.sum(axis=1)
            ll += log_lik.sum()

        if verbose:
            print(f"iter {it+1:2d}: complete-data LL = {ll:.2f}")
        if np.abs(ll - last_ll) < tol:
            break
        last_ll = ll

    return clf_T, regs_Y, s2, T, last_ll


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
        d_x: int | None = None,
        d_y: int | None = None,
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
        self.log_likelihood = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray, T_obs: np.ndarray) -> "EMModel":
        """Fit the EM model on the provided data."""

        (
            self.clf_T,
            self.regs_Y,
            self.sigma2,
            self.T_imputed,
            self.log_likelihood,
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
    def predict_treatment_proba(self, Z: np.ndarray) -> np.ndarray:
        """Return ``p(T | X,Y)`` using the learned classifier."""

        if self.clf_T is None:
            raise ValueError("Model is not fitted")
        return self.clf_T.predict_proba(Z)

    # ------------------------------------------------------------------
    def predict_outcome(self, X: np.ndarray, t: int) -> np.ndarray:
        """Predict the outcome for treatment ``t``."""

        if self.regs_Y is None:
            raise ValueError("Model is not fitted")
        return self.regs_Y[t].predict(X)


__all__ = ["em_learn", "EMModel"]
