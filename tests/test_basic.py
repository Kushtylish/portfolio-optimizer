import numpy as np
import pandas as pd
from src.optimize import max_sharpe_weights, min_variance_weights
from src.estimators import estimate_cov
from src.metrics import metrics


def test_weights_sum_to_one():
    np.random.seed(0)
    rets = pd.DataFrame(np.random.randn(500, 5) * 0.01)
    cov = estimate_cov(rets, "ledoit")
    mu = rets.mean().values * 252
    w = max_sharpe_weights(mu, cov)
    assert abs(w.sum() - 1) < 1e-6
    assert (w >= -1e-8).all()
    assert (w <= 1 + 1e-8).all()


def test_min_var_equal_weight_on_diag_cov():
    cov = np.eye(3) * 0.04
    w = min_variance_weights(cov)
    assert np.allclose(w, 1 / 3, atol=1e-3)


def test_ledoit_cov_is_psd():
    np.random.seed(1)
    rets = pd.DataFrame(np.random.randn(300, 8) * 0.01)
    cov = estimate_cov(rets, "ledoit")
    eigvals = np.linalg.eigvalsh(cov)
    assert (eigvals > 0).all()


def test_metrics_known_series():
    r = pd.Series([0.001] * 252)
    m = metrics(r)
    assert m["AnnVol"] < 1e-6