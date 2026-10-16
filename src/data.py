import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "TLT", "GLD", "VNQ", "DBC"]
START = "2010-01-01"
END = "2024-12-31"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(tickers=TICKERS, start=START, end=END, cache=True):
    DATA_DIR.mkdir(exist_ok=True)
    cache_file = DATA_DIR / "prices.csv"

    if cache and cache_file.exists():
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        prices = yf.download(tickers, start=start, end=end, auto_djust=True)["Close"]
        prices = prices.dropna(how="all")
        try:
            prices.to_csv(cache_file)
        except OSError:
            pass
    returns = prices.pct_change().dropna()
    return prices, returns