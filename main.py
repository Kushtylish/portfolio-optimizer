import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.data import load_data
from src.backtest import backtest
from src.metrics import metrics


def run():
    _, returns = load_data()

    configs = [
        ("max_sharpe", "sample"),
        ("max_sharpe", "ledoit"),
        ("min_var", "sample"),
        ("min_var", "ledoit"),
    ]

    results = {}
    curves = {}
    for method, est in configs:
        ret, w, to = backtest(returns, method=method, estimator=est)
        name = f"{method}_{est}"
        results[name] = metrics(ret)
        curves[name] = (1 + ret).cumprod()

    summary = pd.DataFrame(results).T
    print(summary.round(3))

    ax = pd.DataFrame(curves).plot(figsize=(12, 6), title="Cumulative Returns")
    ax.set_ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig("results/cumulative_returns.png", dpi=150)
    print("\nSaved plot to results/cumulative_returns.png")

    summary.to_csv("results/metrics.csv")


if __name__ == "__main__":
    run()