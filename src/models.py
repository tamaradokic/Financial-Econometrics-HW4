"""
models.py
---------
Econometric models for realized variance forecasting.

Implements:
  - HAR  (Heterogeneous Autoregressive, Corsi 2009)
  - AR(1)
  - Random Walk

All models share a common fit/predict interface used by forecasting.py.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# ── HAR model ────────────────────────────────────────────────────────────────

@dataclass
class HARModel:
    """
    HAR model for lnRV:
        y_t = β0 + β1·y_{t-1} + β2·ȳ^(5)_{t-1} + β3·ȳ^(22)_{t-1} + ε_t

    where ȳ^(n)_{t-1} = (1/n) Σ_{i=1}^{n} y_{t-i}

    Estimated by OLS (HAR is linear in parameters).
    """
    params_: np.ndarray = field(default=None, init=False, repr=False)
    param_names: list = field(default_factory=lambda: ["const", "β_d", "β_w", "β_m"],
                              init=False)

    def _build_X(self, lnrv: np.ndarray) -> np.ndarray:
        """Build [1, y_{t-1}, ȳ^5_{t-1}, ȳ^22_{t-1}] feature matrix."""
        T = len(lnrv)
        y_d = np.roll(lnrv, 1)
        y_w = np.array([lnrv[max(0, i-5):i].mean() if i >= 5 else np.nan
                        for i in range(T)])
        y_m = np.array([lnrv[max(0, i-22):i].mean() if i >= 22 else np.nan
                        for i in range(T)])
        X = np.column_stack([np.ones(T), y_d, y_w, y_m])
        return X

    def fit(self, lnrv: pd.Series) -> "HARModel":
        """Fit on a lnRV series. Drops first 22 observations (warm-up)."""
        y = lnrv.values
        X = self._build_X(y)
        # first 22 rows have NaN in y_m
        mask = ~np.isnan(X).any(axis=1)
        res = OLS(y[mask], X[mask]).fit()
        self.params_ = res.params
        self.result_ = res
        return self

    def predict_one(self, history: np.ndarray) -> float:
        """
        One-step-ahead forecast using the last elements of history.
        history : 1-D array of past lnRV values (at least 22 elements).
        """
        y_d = history[-1]
        y_w = history[-5:].mean()
        y_m = history[-22:].mean()
        x = np.array([1.0, y_d, y_w, y_m])
        return float(x @ self.params_)

    def summary_table(self) -> pd.DataFrame:
        """Return parameter estimates with std errors and t-stats."""
        return pd.DataFrame({
            "Estimate": self.result_.params,
            "Std Err":  self.result_.bse,
            "t-stat":   self.result_.tvalues,
            "p-value":  self.result_.pvalues,
        }, index=self.param_names)


# ── AR(1) model ──────────────────────────────────────────────────────────────

@dataclass
class AR1Model:
    """
    AR(1):  y_t = α + φ·y_{t-1} + ε_t
    Estimated by OLS.
    """
    params_: np.ndarray = field(default=None, init=False, repr=False)

    def fit(self, lnrv: pd.Series) -> "AR1Model":
        y = lnrv.values
        X = add_constant(y[:-1])
        res = OLS(y[1:], X).fit()
        self.params_ = res.params   # [intercept, phi]
        return self

    def predict_one(self, history: np.ndarray) -> float:
        return float(self.params_[0] + self.params_[1] * history[-1])


# ── Random Walk ───────────────────────────────────────────────────────────────

class RandomWalkModel:
    """
    Naive benchmark: ŷ_{t+1} = y_t
    No parameters to estimate.
    """
    def fit(self, lnrv: pd.Series) -> "RandomWalkModel":
        return self

    def predict_one(self, history: np.ndarray) -> float:
        return float(history[-1])


# ── HAR coefficient table across all stocks ──────────────────────────────────

def fit_har_all_stocks(lnrv_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Fit HAR on the full sample for every stock.

    Returns a DataFrame where rows = tickers and columns = [β0, β_d, β_w, β_m, R²].
    """
    rows = []
    for ticker in lnrv_panel.columns:
        series = lnrv_panel[ticker].dropna()
        if len(series) < 50:
            continue
        model = HARModel().fit(series)
        row = {
            "Ticker":  ticker,
            "β0":      round(model.params_[0], 4),
            "β_d":     round(model.params_[1], 4),
            "β_w":     round(model.params_[2], 4),
            "β_m":     round(model.params_[3], 4),
            "R²":      round(model.result_.rsquared, 4),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("Ticker")
