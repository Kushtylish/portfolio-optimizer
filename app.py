import streamlit as st
import pandas as pd
from src.data import load_data
from src.backtest import backtest
from src.metrics import metrics

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimizer: Sample vs Ledoit-Wolf Covariance")
st.caption(
    "Rolling backtest comparing sample covariance and Ledoit-Wolf shrinkage "
    "in max-Sharpe and min-variance optimization."
)


@st.cache_data
def get_returns():
    _, r = load_data()
    return r


@st.cache_data
def run_backtests(window, rebalance, cost):
    returns = get_returns()
    configs = [
        ("max_sharpe", "sample"),
        ("max_sharpe", "ledoit"),
        ("min_var", "sample"),
        ("min_var", "ledoit"),
    ]
    rows = {}
    curves = {}
    for method, est in configs:
        ret, _, to = backtest(
            returns,
            method=method,
            estimator=est,
            window=window,
            rebalance=rebalance,
            cost=cost,
        )
        name = f"{method}_{est}"
        m = metrics(ret)
        m["Turnover"] = to.mean()
        rows[name] = m
        curves[name] = (1 + ret).cumprod()
    return pd.DataFrame(rows).T, pd.DataFrame(curves)


with st.sidebar:
    st.header("Parameters")
    window = st.slider("Training window (days)", 126, 504, 252, 21)
    rebalance = st.slider("Rebalance frequency (days)", 5, 63, 21, 1)
    cost_bps = st.slider("Cost per unit turnover (bps)", 0, 20, 5)
    cost = cost_bps / 10000

summary, curves = run_backtests(window, rebalance, cost)

st.subheader("Performance metrics")
st.dataframe(summary.round(4), use_container_width=True)

st.subheader("Cumulative returns")
st.line_chart(curves)

st.markdown(
    """
    **Notes:**
    - All strategies are long-only, fully invested, rebalanced on the selected frequency.
    - Costs are charged per unit of turnover, so high-turnover strategies are penalized.
    - Data: 10 liquid ETFs, 2010–2024, via yfinance.
    """
)