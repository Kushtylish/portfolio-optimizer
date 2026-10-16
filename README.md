# Portfolio Optimizer: Sample vs Ledoit-Wolf Covariance
[**🔗 Live demo →**](https://portfolio-optimizer-zmhf6hjp2pxehfrpfluakn.streamlit.app/)

## Problem

Mean-variance optimization is notoriously unstable out-of-sample because the
sample covariance matrix is a poor estimator when the number of assets is not
much smaller than the number of observations. This project quantifies that
instability and tests Ledoit-Wolf shrinkage as a fix, in a realistic rolling
backtest with transaction costs.

## Data

10 liquid ETFs representing major asset classes:

- Equities: SPY, QQQ, IWM, EFA, EEM
- Fixed income: AGG, TLT
- Real assets: GLD, VNQ, DBC

Daily adjusted close, 2010-01-01 to 2024-12-31, via yfinance.

## Method

- **Rolling backtest**: 252-day training window, rebalanced every 21 days
- **Optimizers**: max Sharpe, minimum variance
- **Covariance estimators**: sample covariance, Ledoit-Wolf shrinkage
- **Constraints**: long-only, fully invested, max weight 100%
- **Costs**: 5 bps per unit of turnover

## Results

| Strategy | AnnReturn | AnnVol | Sharpe | MaxDD | Avg Daily Turnover |
|---|---|---|---|---|---|
| max_sharpe_sample | 0.051 | 0.114 | 0.451 | -0.255 | 0.021 |
| max_sharpe_ledoit | 0.056 | 0.118 | 0.480 | -0.256 | 0.021 |
| min_var_sample    | 0.024 | 0.046 | 0.529 | -0.168 | 0.002 |
| min_var_ledoit    | 0.027 | 0.049 | 0.547 | -0.174 | 0.003 |

![Cumulative returns](results/cumulative_returns.png)

## Key findings

1. **Ledoit-Wolf shrinkage improves Sharpe in both optimizers**, modestly
   (0.451 → 0.480 for max-Sharpe, 0.529 → 0.547 for min-variance).
2. **Turnover is essentially unchanged** between sample and shrinkage in this
   universe. With only 10 assets and a 252-day window, T >> N, so the sample
   covariance is not the bottleneck. Shrinkage's benefit grows with many assets.
3. **Min-variance dominates max-Sharpe out-of-sample.** Max-Sharpe depends on
   estimated expected returns, which are extremely noisy. Min-variance uses
   only the covariance matrix, which is far more stable. Turnover is 10× lower.
4. **The remaining instability in max-Sharpe comes from means, not covariance.**
   Shrinking the covariance cannot fix a noisy mean estimate.

## Limitations

- No shrinkage on expected returns (James-Stein, Black-Litterman).
- No factor-model covariance (PCA, Fama-French).
- Single backtest path — no bootstrap confidence intervals on Sharpe.
- Flat transaction cost model, not volume-dependent.
- Universe is small (N=10), so sample covariance is better conditioned than in
  a realistic equity universe of 500+ names.

## Next steps

- Add factor-model covariance and compare to Ledoit-Wolf.
- Add James-Stein shrinkage on mean returns.
- Bootstrap Sharpe ratios to test significance.
- Expand universe to 100+ assets to see shrinkage matter more.
- Interactive Streamlit dashboard.

## Reproduce

```bash
pip install -r requirements.txt
python main.py