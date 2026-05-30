"""
rv_construction.py
------------------
Converts 1-minute OHLCV data into daily Realized Variance (RV) series.

Pipeline per stock:
  1. Extract Close prices, drop NaN rows (non-trading minutes)
  2. Resample to 5-minute bars (last close within each 5-min window)
  3. Keep only bars within regular trading hours: 09:30 – 15:55
  4. Compute 5-minute log returns:  r_s = log(P_s / P_{s-1})
  5. Sum squared returns within each trading day → RV_t = Σ r_s²
  6. Log-transform → lnRV_t = log(RV_t)

This follows Andersen et al. (2001) and is the standard in the RV literature.
5-minute sampling is chosen to balance microstructure noise against information
content (Liu et al., 2015).
"""

import numpy as np
import pandas as pd
from typing import Union


# ── constants ────────────────────────────────────────────────────────────────
RESAMPLE_FREQ = "5min"          # sub-sampling frequency
SESSION_START = "09:30"         # NYSE open
SESSION_END   = "15:55"         # last 5-min bar start before close at 16:00


def compute_rv(stock_df: pd.DataFrame, ticker: str = "") -> pd.Series:
    """
    Compute daily Realized Variance from 1-minute OHLCV data.

    Parameters
    ----------
    stock_df : DataFrame with DatetimeIndex and a 'Close' column
    ticker   : optional name for the resulting Series

    Returns
    -------
    pd.Series  daily RV indexed by business date, named by ticker
    """
    # Step 1 – extract close prices and drop NaN (non-trading) rows
    close = stock_df["Close"].dropna()

    # Step 2 – resample to 5-minute bars (last close price in window)
    close_5min = close.resample(RESAMPLE_FREQ).last().dropna()

    # Step 3 – keep only regular trading session bars
    close_5min = close_5min.between_time(SESSION_START, SESSION_END)

    # Step 4 – compute 5-minute log returns within each day
    # Group by date and compute within-day returns only (drops the first
    # bar of each day, which would otherwise capture the overnight gap).
    log_prices = np.log(close_5min)

    def _intraday_returns(group):
        return group.diff().dropna()

    log_returns = (
        log_prices
        .groupby(log_prices.index.date, group_keys=False)
        .apply(_intraday_returns)
        .dropna()
    )

    # Step 5 – daily RV = sum of squared intraday log returns
    rv = (log_returns ** 2).resample("B").sum()

    # Remove days where no intraday data was present (sum would be 0)
    rv = rv[rv > 1e-12]

    rv.name = ticker or "RV"
    rv.index.name = "date"
    return rv


def compute_lnrv(rv: pd.Series) -> pd.Series:
    """Log-transform RV. Handles any non-positive values safely."""
    lnrv = np.log(rv.clip(lower=1e-20))
    lnrv.name = (rv.name or "RV") + "_ln"
    return lnrv


def build_rv_panel(stock_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a (dates × tickers) panel of daily RV for all stocks.

    Parameters
    ----------
    stock_data : dict {ticker: DataFrame} from data_loader.load_all_stocks()

    Returns
    -------
    DataFrame  shape (T, N), columns = tickers, index = business dates
    """
    series = {}
    for ticker, df in stock_data.items():
        try:
            series[ticker] = compute_rv(df, ticker)
        except Exception as e:
            print(f"  WARNING: could not compute RV for {ticker}: {e}")
    panel = pd.DataFrame(series)
    panel.index.name = "date"
    return panel


def build_lnrv_panel(rv_panel: pd.DataFrame) -> pd.DataFrame:
    """Log-transform a full RV panel."""
    return np.log(rv_panel.clip(lower=1e-20))


def build_har_features(lnrv: pd.Series) -> pd.DataFrame:
    """
    Construct the HAR feature matrix for a single lnRV series.

    Features:
      y_d   = lnRV_{t-1}          (daily lag)
      y_w   = mean(lnRV_{t-5:t-1}) (weekly average, 5 days)
      y_m   = mean(lnRV_{t-22:t-1})(monthly average, 22 days)
      y     = lnRV_t               (target)

    Returns a DataFrame with no NaN rows (first 22 obs dropped).
    """
    name = lnrv.name or "lnRV"
    df = pd.DataFrame({"y": lnrv})
    df["y_d"] = lnrv.shift(1)
    df["y_w"] = lnrv.shift(1).rolling(5).mean()
    df["y_m"] = lnrv.shift(1).rolling(22).mean()
    return df.dropna()
