# em_baseline.py
# Dempster, Laird & Rubin “Maximum Likelihood from Incomplete Data via the EM Algorithm”, JRSS-B 1977
# link.springer.com
# Nigam et al. “Text Classification from Labeled and Unlabeled Documents using EM”, MLJ 2000
# link.springer.com

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import norm


def em_learn(
    X,
    Y,
    T_obs,  # arrays of shape (n, d), (n,), (n,) -- T_obs may have -1 for "missing"
    n_treatments,
    max_iter=20,
    tol=1e-4,
    C_logreg=1.0,  # regularisation for the logistic head
    verbose=False,
):
    """Return:
    - clf_T      : LogisticRegression modelling  p(T | X)
    - regs_Y[t]  : list of LinearRegression modelling E[Y | X, T=t]
    - sigma2[t]  : residual variance per treatment
    - T_imputed  : np.ndarray of shape (n,) with hard EM labels
    """
    # ----- split labelled vs unlabelled ----------------------------------
    labelled = T_obs != -1
    unlabelled = ~labelled
    T = T_obs.copy()

    # ----- helpers --------------------------------------------------------
    def fit_logreg(X_, T_):
        lr = LogisticRegression(
            multi_class="multinomial",
            max_iter=200,
            C=C_logreg,
            solver="lbfgs",
        ).fit(X_, T_)
        return lr

    def fit_regressions(X_, Y_, T_):
        regs, sig2 = [], []
        for t in range(n_treatments):
            mask = T_ == t
            # If a treatment is absent in current pseudo-labels, keep a dummy regressor
            if mask.sum() < 2:
                regs.append(
                    LinearRegression().fit(np.zeros((1, X_.shape[1])), [Y_.mean()])
                )
                sig2.append(Y_.var() + 1e-6)
                continue
            r = LinearRegression().fit(X_[mask], Y_[mask])
            resid = Y_[mask] - r.predict(X_[mask])
            regs.append(r)
            sig2.append(resid.var() + 1e-6)  # avoid zero variance
        return regs, np.array(sig2)

    # ----- initialisation: supervise only on labelled data ---------------
    clf_T = fit_logreg(X[labelled], T[labelled])
    regs_Y, s2 = fit_regressions(X[labelled], Y[labelled], T[labelled])

    last_ll = -np.inf
    for it in range(max_iter):
        # ===== E-step: impute missing T via MAP ===========================
        log_post = np.zeros((unlabelled.sum(), n_treatments))
        p_t_given_x = clf_T.predict_proba(X[unlabelled])  # shape (n_U, K)
        y_u = Y[unlabelled]

        for t in range(n_treatments):
            mu = regs_Y[t].predict(X[unlabelled])
            log_lik = norm.logpdf(y_u, loc=mu, scale=np.sqrt(s2[t]))
            log_post[:, t] = np.log(p_t_given_x[:, t] + 1e-12) + log_lik

        T_hat = log_post.argmax(axis=1)
        T[unlabelled] = T_hat

        # ===== M-step: refit heads on (labelled + pseudo-labelled) =======
        clf_T = fit_logreg(X, T)
        regs_Y, s2 = fit_regressions(X, Y, T)

        # ===== Convergence check: complete-data log-likelihood ===========
        ll = 0.0
        # p(T|X) term
        ll += clf_T.predict_log_proba(X[np.arange(len(T)), T]).sum()
        # p(Y|X,T) term
        for t in range(n_treatments):
            mask = T == t
            mu = regs_Y[t].predict(X[mask])
            ll += norm.logpdf(Y[mask], loc=mu, scale=np.sqrt(s2[t])).sum()

        if verbose:
            print(f"iter {it+1:2d}: complete-data LL = {ll:.2f}")
        if np.abs(ll - last_ll) < tol:
            break
        last_ll = ll

    return clf_T, regs_Y, s2, T
