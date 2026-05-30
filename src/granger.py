"""
granger.py
----------
Pairwise Granger Causality tests across all DJIA stocks.
"""

import numpy as np
import pandas as pd
import itertools

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import plotly.graph_objects as go


# ── helpers ───────────────────────────────────────────────────────────────────

def _mfe(actual, forecast):
    return float(np.mean(forecast - actual))

def _rmse(actual, forecast):
    return float(np.sqrt(np.mean((forecast - actual) ** 2)))


# ── pairwise Granger tests ────────────────────────────────────────────────────

def run_pairwise_granger(
    lnrv_panel: pd.DataFrame,
    max_lag: int = 10,
    significance: float = 0.05,
) -> pd.DataFrame:
    """
    Test all ordered pairs (X → Y) for Granger causality.
    Returns DataFrame with columns [From, To, p_value, Best_Lag, Significant].
    """
    tickers = lnrv_panel.columns.tolist()
    rows = []
    pairs = list(itertools.permutations(tickers, 2))
    n = len(pairs)
    print(f"Running {n} Granger causality tests …", flush=True)

    for i, (x_ticker, y_ticker) in enumerate(pairs):
        if i % 100 == 0:
            print(f"  {i}/{n} pairs done …", flush=True)

        data = lnrv_panel[[y_ticker, x_ticker]].dropna()
        if len(data) < 50:
            continue

        try:
            results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            pvals = {lag: res[0]["ssr_ftest"][1] for lag, res in results.items()}
            best_lag = min(pvals, key=pvals.get)
            min_p = pvals[best_lag]
            rows.append({
                "From":        x_ticker,
                "To":          y_ticker,
                "p_value":     round(min_p, 4),
                "Best_Lag":    best_lag,
                "Significant": min_p < significance,
            })
        except Exception:
            pass

    df = pd.DataFrame(rows)
    df = df.sort_values("p_value").reset_index(drop=True)
    return df


def granger_heatmap(granger_df: pd.DataFrame,
                    tickers: list) -> go.Figure:
    """Heatmap of Granger causality p-values (From → To)."""
    mat = pd.DataFrame(np.nan, index=tickers, columns=tickers)
    for _, row in granger_df.iterrows():
        mat.loc[row["From"], row["To"]] = row["p_value"]

    fig = go.Figure(go.Heatmap(
        z=mat.values,
        x=mat.columns.tolist(),
        y=mat.index.tolist(),
        colorscale="RdYlGn_r",
        zmin=0, zmax=0.1,
        colorbar=dict(title="p-value"),
    ))
    fig.update_layout(
        title="Pairwise Granger Causality p-values (From → To)",
        xaxis_title="To (Y)", yaxis_title="From (X)",
        height=700, width=750,
        template="plotly_white",
        margin=dict(l=80, r=20, t=60, b=80),
    )
    return fig


def top_granger_pairs(granger_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return the top-n most significant Granger causality pairs."""
    sig = granger_df[granger_df["Significant"]].head(n)
    return sig[["From", "To", "p_value", "Best_Lag"]].copy()


# ── VAR forecasts for Granger-significant pairs ───────────────────────────────

def var_forecast_pair(
    lnrv_panel: pd.DataFrame,
    x_ticker: str,
    y_ticker: str,
    max_lag: int = 5,
) -> dict:
    """Fit VAR once on training set, forecast over test set."""
    data    = lnrv_panel[[y_ticker, x_ticker]].dropna()
    arr_y   = data[y_ticker].values
    T       = len(arr_y)
    n_test  = T // 2
    n_train = T - n_test
    actual  = arr_y[n_train:]
    var_forecasts = np.full(n_test, np.nan)

    try:
        res  = VAR(data.iloc[:n_train]).fit(maxlags=max_lag, ic="bic")
        k_ar = res.k_ar
    except Exception:
        return {"pair": f"{x_ticker}→{y_ticker}", "var_rmse": np.nan,
                "var_mfe": np.nan, "forecasts": var_forecasts, "actual": actual}

    for i in range(n_test):
        try:
            history = data.iloc[max(0, n_train + i - k_ar): n_train + i].values
            var_forecasts[i] = res.forecast(history, steps=1)[0, 0]
        except Exception:
            var_forecasts[i] = arr_y[n_train + i - 1]

    return {
        "pair":      f"{x_ticker}→{y_ticker}",
        "var_rmse":  round(_rmse(actual, var_forecasts), 6),
        "var_mfe":   round(_mfe(actual, var_forecasts), 6),
        "forecasts": var_forecasts,
        "actual":    actual,
    }


def run_var_for_significant_pairs(
    lnrv_panel: pd.DataFrame,
    granger_df: pd.DataFrame,
    har_results: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Run VAR for top Granger pairs and compare against HAR benchmark."""
    pairs = top_granger_pairs(granger_df, n=top_n)
    rows  = []

    for _, row in pairs.iterrows():
        x, y = row["From"], row["To"]
        print(f"  VAR forecast: {x} → {y}", flush=True)
        res = var_forecast_pair(lnrv_panel, x, y)

        har_row = har_results[
            (har_results["Ticker"] == y) & (har_results["Model"] == "HAR")
        ]
        har_rmse_val = har_row["RMSE"].values[0] if len(har_row) else np.nan

        rows.append({
            "Pair":         res["pair"],
            "VAR RMSE":     res["var_rmse"],
            "HAR RMSE (Y)": round(har_rmse_val, 6),
            "Improvement":  round(har_rmse_val - res["var_rmse"], 6),
            "p-value":      row["p_value"],
        })

    df = pd.DataFrame(rows)
    df["Better?"] = df["Improvement"] > 0
    return df.sort_values("Improvement", ascending=False)
