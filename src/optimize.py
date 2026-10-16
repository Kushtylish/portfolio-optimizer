import numpy as np
from scipy.optimize import minimize


def max_sharpe_weights(mu, cov, rf=0.0, max_weight=1.0):
    n = len(mu)

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        if vol < 1e-12:
            return 1e6
        return -(ret - rf) / vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n
    w0 = np.ones(n) / n

    res = minimize(
        neg_sharpe,
        w0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-9},
    )
    return res.x if res.success else w0


def min_variance_weights(cov, max_weight=1.0):
    n = cov.shape[0]

    def var(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n
    w0 = np.ones(n) / n

    res = minimize(var, w0, bounds=bounds, constraints=constraints, method="SLSQP")
    return res.x if res.success else w0