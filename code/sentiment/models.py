"""Linear text classifiers implemented from scratch with NumPy.

* :class:`MultinomialNB`      - classic generative baseline.
* :class:`LogisticRegression` - L2-regularised, trained with a hand-written
                                 L-BFGS optimiser (same objective as
                                 scikit-learn's ``LogisticRegression(C=...)``).
* :class:`NBLogisticRegression` - the "NB-SVM"/NB-LR model of Wang & Manning
                                 (2012): binarised n-gram presence features are
                                 scaled by naive-Bayes log-count ratios before a
                                 discriminative linear model is fitted, and the
                                 learned weights are interpolated towards their
                                 mean to improve generalisation.

All models expect a :class:`~sentiment.sparse.CSRMatrix` ``X`` and a label
vector ``y`` with values in {0, 1}; ``decision_function`` returns a real score
whose sign is the prediction and ``predict_proba`` returns P(y = 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sparse import CSRMatrix


# ----------------------------------------------------------------------------
# Optimiser
# ----------------------------------------------------------------------------
def _lbfgs(fun_grad, x0: np.ndarray, max_iter: int = 300, history: int = 10, gtol: float = 1e-5,
           ftol: float = 1e-9, verbose: bool = False) -> tuple[np.ndarray, list[float]]:
    """Limited-memory BFGS with an Armijo backtracking line search.

    ``fun_grad(x)`` must return ``(loss, gradient)``.  Returns the minimiser and
    the loss trajectory (useful for convergence plots).
    """
    x = x0.astype(np.float64).copy()
    f, g = fun_grad(x)
    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    rho_hist: list[float] = []
    trace = [float(f)]
    for it in range(max_iter):
        # Two-loop recursion to compute the search direction d = -H g.
        q = g.copy()
        alphas = []
        for s, y, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
            a = rho * s.dot(q)
            alphas.append(a)
            q -= a * y
        if s_hist:
            gamma = s_hist[-1].dot(y_hist[-1]) / y_hist[-1].dot(y_hist[-1])
        else:
            gamma = 1.0 / max(np.linalg.norm(g), 1.0)
        r = gamma * q
        for (s, y, rho), a in zip(zip(s_hist, y_hist, rho_hist), reversed(alphas)):
            b = rho * y.dot(r)
            r += s * (a - b)
        d = -r
        slope = g.dot(d)
        if slope >= 0:  # not a descent direction: reset memory and use steepest descent
            s_hist.clear(); y_hist.clear(); rho_hist.clear()
            d = -g
            slope = g.dot(d)
        # Backtracking (Armijo) line search.
        step = 1.0
        while True:
            x_new = x + step * d
            f_new, g_new = fun_grad(x_new)
            if f_new <= f + 1e-4 * step * slope or step < 1e-10:
                break
            step *= 0.5
        s = x_new - x
        y = g_new - g
        sy = s.dot(y)
        if sy > 1e-12:
            s_hist.append(s); y_hist.append(y); rho_hist.append(1.0 / sy)
            if len(s_hist) > history:
                s_hist.pop(0); y_hist.pop(0); rho_hist.pop(0)
        rel_decrease = (f - f_new) / max(abs(f), 1e-12)
        x, f, g = x_new, f_new, g_new
        trace.append(float(f))
        if verbose and (it % 10 == 0):
            print(f"    iter {it:4d}  loss {f:.6f}  |g|_inf {np.abs(g).max():.2e}")
        if np.abs(g).max() < gtol or (0 <= rel_decrease < ftol):
            break
    return x, trace


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
@dataclass
class MultinomialNB:
    """Multinomial naive Bayes with Laplace/Lidstone smoothing ``alpha``."""

    alpha: float = 1.0
    feature_log_prob_: np.ndarray | None = field(default=None, repr=False)
    class_log_prior_: np.ndarray | None = field(default=None, repr=False)

    def fit(self, X: CSRMatrix, y: np.ndarray) -> "MultinomialNB":
        y = np.asarray(y)
        n_features = X.shape[1]
        logp = np.zeros((2, n_features))
        prior = np.zeros(2)
        for c in (0, 1):
            mask = y == c
            counts = X.column_sums(mask) + self.alpha
            logp[c] = np.log(counts) - np.log(counts.sum())
            prior[c] = np.log(mask.mean())
        self.feature_log_prob_ = logp
        self.class_log_prior_ = prior
        return self

    def decision_function(self, X: CSRMatrix) -> np.ndarray:
        w = self.feature_log_prob_[1] - self.feature_log_prob_[0]
        b = self.class_log_prior_[1] - self.class_log_prior_[0]
        return X.dot(w) + b

    def predict_proba(self, X: CSRMatrix) -> np.ndarray:
        return _sigmoid(self.decision_function(X))

    def predict(self, X: CSRMatrix) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(np.int64)


@dataclass
class LogisticRegression:
    """Binary logistic regression, objective = mean log-loss + (1 / (2 C n)) ||w||^2.

    ``C`` therefore has exactly the same meaning as in scikit-learn.
    """

    C: float = 1.0
    max_iter: int = 300
    fit_intercept: bool = True
    verbose: bool = False
    coef_: np.ndarray | None = field(default=None, repr=False)
    intercept_: float = 0.0
    loss_trace_: list[float] = field(default_factory=list, repr=False)

    def _unpack(self, theta: np.ndarray, n_features: int) -> tuple[np.ndarray, float]:
        if self.fit_intercept:
            return theta[:n_features], float(theta[n_features])
        return theta, 0.0

    def fit(self, X: CSRMatrix, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "LogisticRegression":
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        sw = sw / sw.sum()
        lam = 1.0 / (self.C * n)
        t = 2.0 * y - 1.0  # labels in {-1, +1}
        # Reuse one float64 snapshot only for this fit. Keeping it local avoids
        # retaining a second data array or stale values after X.data is changed.
        data64 = X.data.astype(np.float64)

        def fun_grad(theta: np.ndarray):
            w, b = self._unpack(theta, p)
            z = X.dot(w, data64=data64) + b
            margin = t * z
            # log(1 + exp(-margin)) computed stably
            loss = np.sum(sw * np.logaddexp(0.0, -margin)) + 0.5 * lam * w.dot(w)
            # d loss / d z = -t * sigmoid(-margin)
            dz = -t * _sigmoid(-margin) * sw
            gw = X.rdot(dz, data64=data64) + lam * w
            if self.fit_intercept:
                return loss, np.concatenate([gw, [dz.sum()]])
            return loss, gw

        theta0 = np.zeros(p + (1 if self.fit_intercept else 0))
        theta, trace = _lbfgs(fun_grad, theta0, max_iter=self.max_iter, verbose=self.verbose)
        self.coef_, self.intercept_ = self._unpack(theta, p)
        self.loss_trace_ = trace
        return self

    def decision_function(self, X: CSRMatrix) -> np.ndarray:
        return X.dot(self.coef_) + self.intercept_

    def predict_proba(self, X: CSRMatrix) -> np.ndarray:
        return _sigmoid(self.decision_function(X))

    def predict(self, X: CSRMatrix) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(np.int64)


@dataclass
class NBLogisticRegression:
    """NB-LR (Wang & Manning, 2012) with weight interpolation ``beta``.

    1. Binarise the count matrix (presence/absence of each n-gram).
    2. Compute the smoothed log-count ratio
       ``r = log((p / ||p||_1) / (q / ||q||_1))`` where ``p``/``q`` are the
       feature counts in the positive/negative class plus ``alpha``.
    3. Train logistic regression on ``X_bin * r``.
    4. Interpolate: ``w <- (1 - beta) * mean(|w|) + beta * w`` (beta = 1 is
       plain LR on NB-scaled features, beta = 0 is pure NB-style weighting).
    """

    C: float = 1.0
    alpha: float = 1.0
    beta: float = 0.25
    max_iter: int = 300
    r_: np.ndarray | None = field(default=None, repr=False)
    lr_: LogisticRegression | None = field(default=None, repr=False)
    coef_: np.ndarray | None = field(default=None, repr=False)
    intercept_: float = 0.0

    def _transform(self, X: CSRMatrix) -> CSRMatrix:
        return X.binarize().scale_columns(self.r_)

    def fit(self, X: CSRMatrix, y: np.ndarray) -> "NBLogisticRegression":
        y = np.asarray(y)
        Xb = X.binarize()
        p = Xb.column_sums(y == 1) + self.alpha
        q = Xb.column_sums(y == 0) + self.alpha
        self.r_ = (np.log(p / p.sum()) - np.log(q / q.sum())).astype(np.float32)
        self.lr_ = LogisticRegression(C=self.C, max_iter=self.max_iter).fit(Xb.scale_columns(self.r_), y)
        w = self.lr_.coef_
        w_bar = np.abs(w).mean()
        self.coef_ = (1.0 - self.beta) * w_bar + self.beta * w
        self.intercept_ = self.lr_.intercept_
        return self

    def decision_function(self, X: CSRMatrix) -> np.ndarray:
        return self._transform(X).dot(self.coef_) + self.intercept_

    def predict_proba(self, X: CSRMatrix) -> np.ndarray:
        return _sigmoid(self.decision_function(X))

    def predict(self, X: CSRMatrix) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(np.int64)
