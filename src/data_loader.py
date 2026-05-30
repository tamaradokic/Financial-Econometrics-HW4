"""
data_loader.py
--------------
Loads raw 1-minute OHLCV CSV files for each DJIA stock and attaches the
shared intraday timestamp index from HFdaylistdates.csv.

Each CSV has 5 columns (Open, High, Low, Close, Volume) with no header.
The timestamp file (HFdaylistdates.csv) has one row per minute, aligned
row-for-row with the price CSVs. Both span Jan 2003 – Mar 2022.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TIMESTAMP_FILE = DATA_DIR / "HFdaylistdates.csv"

# All DJIA tickers present in the data folder
TICKERS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
    "DIS", "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ",
    "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG",
    "TRV", "UNH", "V", "VZ", "WBA", "WMT",
]


def load_timestamps() -> pd.DatetimeIndex:
    """
    Load the shared 1-minute timestamp index.
    The header row is itself a timestamp, so we read with header=None.
    """
    ts = pd.read_csv(TIMESTAMP_FILE, header=None).iloc[:, 0]
    return pd.to_datetime(ts.values)


def load_stock(ticker: str, timestamps: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """
    Load 1-minute OHLCV data for a single ticker.

    Parameters
    ----------
    ticker     : stock symbol (e.g. 'AAPL')
    timestamps : pre-loaded DatetimeIndex; loaded lazily if None

    Returns
    -------
    DataFrame with columns [Open, High, Low, Close, Volume]
    and a DatetimeIndex. NaN rows (non-trading minutes) are kept
    so the index aligns perfectly with the timestamp file.
    """
    if timestamps is None:
        timestamps = load_timestamps()

    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data file for {ticker} at {path}")

    df = pd.read_csv(
        path,
        header=None,
        names=["Open", "High", "Low", "Close", "Volume"],
        dtype=float,
    )
    df.index = timestamps
    df.index.name = "datetime"
    return df


def load_all_stocks(tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """
    Load all tickers into a dict {ticker: DataFrame}.
    Uses a single timestamp load for efficiency.
    """
    tickers = tickers or TICKERS
    timestamps = load_timestamps()
    print(f"Loading {len(tickers)} stocks …", flush=True)
    data = {}
    for t in tickers:
        try:
            data[t] = load_stock(t, timestamps)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
    print(f"  Loaded {len(data)} stocks successfully.")
    return data
