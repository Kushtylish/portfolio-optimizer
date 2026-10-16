import numpy as np
from sklearn.covariance import LedoitWolf


def estimate_cov(returns, method="ledoit"):
    """Return annualized covariance matrix."""
    if method == "sample":
        cov = returns.cov().values
    elif method == "ledoit":
        cov = LedoitWolf().fit(returns.values).covariance_
    else:
        raise ValueError(f"Unknown method: {method}")
    return cov * 252