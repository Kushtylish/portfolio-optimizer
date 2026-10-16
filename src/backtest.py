import numpy as np
import pandas as pd
from src.estimators import estimate_cov
from src.optimize import max_sharpe_weights, min_variance_weights


def backtest(
    returns,
    method="max_sharpe",
    estimator="ledoit",
    window=252,
    rebalance=21,
    cost=0.0005,
):
    dates = returns.index
    weights = pd.DataFrame(index=dates, columns=returns.columns, dtype=float)

    for i in range(window, len(dates), rebalance):
        train = returns.iloc[i - window:i]
        if method == "max_sharpe":
            mu = train.mean().values * 252
            cov = estimate_cov(train, estimator)
            w = max_sharpe_weights(mu, cov)
        elif method == "min_var":
            cov = estimate_cov(train, estimator)
            w = min_variance_weights(cov)
        else:
            raise ValueError(method)
        weights.iloc[i:i + rebalance] = w

    weights = weights.ffill().dropna()
    port_ret = (weights.shift(1) * returns).sum(axis=1).dropna()
    turnover = np.abs(weights - weights.shift(1)).sum(axis=1).dropna()
    port_ret_net = port_ret - turnover * cost

    return port_ret_net, weights, turnover