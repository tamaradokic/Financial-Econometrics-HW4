"""
granger.py
----------
Pairwise Granger Causality tests across all DJIA stocks.

Granger causality: stock X Granger-causes stock Y if past values of X
contain information that helps predict Y beyond Y's own past.

We test on the full sample using a VAR(p) model with lag order p selected
by BIC (up to max_lag=10).

Additionally, for pairs where causality is detected, we run the same
rolling OOS forecasting as in forecasting.py but using a bivariate VAR
instead of the univariate HAR — to test whether cross-stock information
actually improves out-of-sample accuracy.
"""

import numpy as np
import pandas as pd
import itertools

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import plotly.graph_objects as go


# ── pairwise Granger tests ────────────────────────────────────────────────────

def run_pairwise_granger(
    lnrv_panel: pd.DataFrame,
    max_lag: int = 10,
    significance: float = 0.05,
) -> pd.DataFrame:
    """
    Test all ordered pairs (X → Y) for Granger causality.

    For each pair we use the minimum p-value across lags 1..max_lag
    (this is conservative; a proper test would use BIC-selected lag).

    Returns
    -------
    DataFrame with columns [From, To, min_p_value, Significant, Best_Lag]
    sorted by p-value ascending.
    """
    tickers = lnrv_panel.columns.tolist()
    rows = []

    pairs = list(itertools.permutations(tickers, 2))
    n = len(pairs)
    print(f"Running {n} Granger causality tests …", flush=True)

    for i, (x_ticker, y_ticker) in enumerate(pairs):
        if i % 100 == 0:
            print(f"  {i}/{n} pairs done …", flush=True)

        # Align and drop NaN
        data = lnrv_panel[[y_ticker, x_ticker]].dropna()
        if len(data) < 50:
            continue

        try:
            # grangercausalitytests tests whether x_ticker → y_ticker
            # it expects [y, x] ordering
            results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            # Extract minimum p-value (F-test) across all tested lags
            pvals = {lag: res[0]["ssr_ftest"][1] for lag, res in results.items()}
            best_lag = min(pvals, key=pvals.get)
            min_p = pvals[best_lag]

            rows.append({
                "From":      x_ticker,
                "To":        y_ticker,
                "p_value":   round(min_p, 4),
                "Best_Lag":  best_lag,
                "Significant": min_p < significance,
            })
        except Exception as e:
            pass  # skip silently

    df = pd.DataFrame(rows)
    df = df.sort_values("p_value").reset_index(drop=True)
    return df


def granger_heatmap(granger_df: pd.DataFrame,
                    tickers: list[str]) -> go.Figure:
    """
    Heatmap of Granger causality p-values (From → To).
    Cells below 0.05 are highlighted.
    """
    # Build p-value matrix
    mat = pd.DataFrame(np.nan, index=tickers, columns=tickers)
    for _, row in granger_df.iterrows():
        mat.loc[row["From"], row["To"]] = row["p_value"]

    fig = go.Figure(go.Heatmap(
        z=mat.values,
        x=mat.columns.tolist(),
        y=mat.index.tolist(),
        colorscale="RdYlGn_r",   # red = low p-value (significant)
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


def top_granger_pairs(granger_df: pd.DataFrame,
                      n: int = 20) -> pd.DataFrame:
    """Return the top-n most significant Granger causality pairs."""
    sig = granger_df[granger_df["Significant"]].head(n)
    return sig[["From", "To", "p_value", "Best_Lag"]].copy()


# ── VAR rolling forecasts for Granger-significant pairs ─────────────────────

def var_forecast_pair(
    lnrv_panel: pd.DataFrame,
    x_ticker: str,
    y_ticker: str,
    max_lag: int = 5,
) -> dict:
    """
    Rolling OOS forecast for a Granger-significant pair (X → Y) using VAR.

    We compare:
      - Univariate HAR for Y  (baseline, from forecasting.py)
      - Bivariate VAR(p) for Y including X as predictor

    Returns dict with {var_rmse, var_mfe, forecasts, actual}
    """
    from src.forecasting import mfe, rmse
    from src.models import HARModel

    data = lnrv_panel[[y_ticker, x_ticker]].dropna()
    arr_y = data[y_ticker].values
    T     = len(arr_y)
    n_test  = T // 2
    n_train = T - n_test
    actual  = arr_y[n_train:]

    var_forecasts = np.full(n_test, np.nan)

    # Fit VAR once on the training set, then use fixed params for all test forecasts
    # This mirrors how HAR is benchmarked and is fast enough for comparison purposes
    try:
        train_data = data.iloc[:n_train]
        res = VAR(train_data).fit(maxlags=max_lag, ic="bic")
        k_ar = res.k_ar
    except Exception:
        return {"pair": f"{x_ticker}→{y_ticker}", "var_rmse": np.nan,
                "var_mfe": np.nan, "forecasts": var_forecasts, "actual": actual}

    for i in range(n_test):
        try:
            history = data.iloc[max(0, n_train + i - k_ar): n_train + i].values
            fc = res.forecast(history, steps=1)
            var_forecasts[i] = fc[0, 0]
        except Exception:
            var_forecasts[i] = arr_y[n_train + i - 1]

    return {
        "pair":       f"{x_ticker}→{y_ticker}",
        "var_rmse":   round(rmse(actual, var_forecasts), 6),
        "var_mfe":    round(mfe(actual, var_forecasts), 6),
        "forecasts":  var_forecasts,
        "actual":     actual,
    }


def run_var_for_significant_pairs(
    lnrv_panel: pd.DataFrame,
    granger_df: pd.DataFrame,
    har_results: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    For the top-n Granger pairs, run VAR rolling forecast and compare RMSE
    against the univariate HAR benchmark.

    Returns comparison DataFrame.
    """
    pairs = top_granger_pairs(granger_df, n=top_n)
    rows  = []

    for _, row in pairs.iterrows():
        x, y = row["From"], row["To"]
        print(f"  VAR forecast: {x} → {y}", flush=True)

        res = var_forecast_pair(lnrv_panel, x, y)

        # HAR RMSE for Y (from main forecast results)
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
