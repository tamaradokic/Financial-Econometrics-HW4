"""
eda.py
------
Exploratory Data Analysis for RV and lnRV series.

Provides:
  - Descriptive statistics (first 4 moments) for RV and lnRV
  - Augmented Dickey-Fuller stationarity tests
  - Time series plots (RV and lnRV)
  - Sample autocorrelation (ACF) plots for lnRV
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf


# ── descriptive statistics ───────────────────────────────────────────────────

def descriptive_stats(series_dict: dict[str, pd.Series], label: str = "RV") -> pd.DataFrame:
    """
    Compute mean, variance, skewness, and excess kurtosis for each series.

    Parameters
    ----------
    series_dict : {ticker: pd.Series}
    label       : 'RV' or 'lnRV', used for column suffix

    Returns
    -------
    DataFrame with tickers as rows and [Mean, Variance, Skewness, Kurtosis] as cols
    """
    rows = []
    for ticker, s in series_dict.items():
        s = s.dropna()
        rows.append({
            "Ticker": ticker,
            "Mean":     s.mean(),
            "Variance": s.var(),
            "Skewness": stats.skew(s),
            "Kurtosis": stats.kurtosis(s),   # excess kurtosis (normal = 0)
            "N":        len(s),
        })
    df = pd.DataFrame(rows).set_index("Ticker")
    return df.round(4)


# ── ADF stationarity tests ───────────────────────────────────────────────────

def adf_tests(lnrv_dict: dict[str, pd.Series], max_lags: int = 20) -> pd.DataFrame:
    """
    Run augmented Dickey-Fuller test on each lnRV series.

    H0: unit root (non-stationary)
    H1: stationary

    Returns DataFrame with ADF statistic, p-value, and conclusion per ticker.
    The lag length is chosen by AIC (up to max_lags).
    """
    rows = []
    for ticker, s in lnrv_dict.items():
        s = s.dropna()
        result = adfuller(s, maxlag=max_lags, autolag="AIC")
        adf_stat, p_val, used_lags = result[0], result[1], result[2]
        rows.append({
            "Ticker":    ticker,
            "ADF Stat":  round(adf_stat, 4),
            "p-value":   round(p_val, 4),
            "Lags Used": used_lags,
            "Stationary (5%)": "Yes" if p_val < 0.05 else "No",
        })
    return pd.DataFrame(rows).set_index("Ticker")


# ── plotly figures ───────────────────────────────────────────────────────────

def plot_rv_timeseries(rv_series: pd.Series, ticker: str) -> go.Figure:
    """Line chart of daily Realized Variance for one stock."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rv_series.index, y=rv_series.values,
        mode="lines", name=ticker,
        line=dict(color="#2563EB", width=1),
    ))
    fig.update_layout(
        title=f"{ticker} – Daily Realized Variance",
        xaxis_title="Date", yaxis_title="RV",
        template="plotly_white", height=350,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def plot_lnrv_timeseries(lnrv_series: pd.Series, ticker: str) -> go.Figure:
    """Line chart of log-transformed RV for one stock."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lnrv_series.index, y=lnrv_series.values,
        mode="lines", name=f"lnRV({ticker})",
        line=dict(color="#7C3AED", width=1),
    ))
    fig.update_layout(
        title=f"{ticker} – log(Realized Variance)",
        xaxis_title="Date", yaxis_title="lnRV",
        template="plotly_white", height=350,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def plot_acf(lnrv_series: pd.Series, ticker: str, n_lags: int = 40) -> go.Figure:
    """Bar chart of sample autocorrelation function for lnRV."""
    acf_vals, confint = acf(lnrv_series.dropna(), nlags=n_lags, alpha=0.05)
    lags = np.arange(0, n_lags + 1)
    ci_upper = confint[:, 1] - acf_vals
    ci_lower = acf_vals - confint[:, 0]

    fig = go.Figure()
    # Confidence band
    ci = 1.96 / np.sqrt(len(lnrv_series))
    fig.add_hline(y=ci,  line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=-ci, line_dash="dash", line_color="red", opacity=0.5)
    # ACF bars
    fig.add_trace(go.Bar(
        x=lags[1:], y=acf_vals[1:],
        marker_color="#0891B2", name="ACF",
    ))
    fig.update_layout(
        title=f"{ticker} – ACF of lnRV",
        xaxis_title="Lag (days)", yaxis_title="Autocorrelation",
        template="plotly_white", height=350,
        margin=dict(l=50, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


def plot_rv_panel_heatmap(rv_panel: pd.DataFrame) -> go.Figure:
    """
    Heatmap of cross-stock RV correlation matrix.
    Useful for spotting co-movement patterns and motivating Granger tests.
    """
    corr = rv_panel.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 7},
    ))
    fig.update_layout(
        title="Cross-stock lnRV Correlation Matrix",
        height=700, width=750,
        template="plotly_white",
        margin=dict(l=80, r=20, t=60, b=80),
    )
    return fig
