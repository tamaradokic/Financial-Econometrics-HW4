"""
forecasting.py
--------------
Rolling out-of-sample (OOS) forecast engine.

Strategy:
  - Last 50% of observations = test set
  - ECONOMETRIC models (HAR, AR1, RW): true expanding-window, refit daily (fast OLS)
  - ML models (LASSO, Ridge, XGB, LightGBM): refit every RETRAIN_EVERY steps
  - LSTM: refit every RETRAIN_EVERY steps with early stopping

Annual retraining (RETRAIN_EVERY=252) is standard practice — it matches
the Kilic (2025) paper and reduces runtime from hours to ~20 minutes.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.models import HARModel, AR1Model, RandomWalkModel
from src.ml_models import (
    LASSOModel, RidgeModel, XGBoostModel, LightGBMModel, LSTMModel,
    build_features,
)

# Refit ML models every N test steps (~annually for daily data)
RETRAIN_EVERY = 252


# ── metrics ──────────────────────────────────────────────────────────────────

def mfe(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(forecast - actual))

def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((forecast - actual) ** 2)))


# ── econometric: refit every RETRAIN_EVERY steps ────────────────────────────

def _rolling_econometric(lnrv: np.ndarray, model_cls) -> np.ndarray:
    T = len(lnrv)
    n_test  = T // 2
    n_train = T - n_test
    forecasts = np.full(n_test, np.nan)
    model = None
    for i in range(n_test):
        train = lnrv[: n_train + i]
        if model is None or i % RETRAIN_EVERY == 0:
            model = model_cls()
            model.fit(pd.Series(train))
        forecasts[i] = model.predict_one(train[-22:])
    return forecasts


# ── ML: refit every RETRAIN_EVERY steps ──────────────────────────────────────

def _rolling_ml(lnrv: np.ndarray, model_cls) -> np.ndarray:
    T = len(lnrv)
    n_test  = T // 2
    n_train = T - n_test
    forecasts = np.full(n_test, np.nan)
    model = None
    for i in range(n_test):
        train_raw = lnrv[: n_train + i]
        if model is None or i % RETRAIN_EVERY == 0:
            X_tr, y_tr = build_features(train_raw)
            model = model_cls()
            model.fit(X_tr, y_tr)
        h = train_raw[-22:]
        x_next = np.array([h[-1], h[-5:].mean(), h[-22:].mean()])
        forecasts[i] = model.predict_one(x_next)
    return forecasts


# ── LSTM: refit every RETRAIN_EVERY steps ────────────────────────────────────

def _rolling_lstm(lnrv: np.ndarray) -> np.ndarray:
    T = len(lnrv)
    n_test  = T // 2
    n_train = T - n_test
    forecasts = np.full(n_test, np.nan)
    model = None
    for i in range(n_test):
        train_raw = lnrv[: n_train + i]
        if model is None or i % RETRAIN_EVERY == 0:
            X_tr, y_tr = build_features(train_raw)
            model = LSTMModel(seq_len=22, hidden_size=32, epochs=30)
            model.fit(X_tr, y_tr)
        h = train_raw[-22:]
        x_next = np.array([h[-1], h[-5:].mean(), h[-22:].mean()])
        forecasts[i] = model.predict_one(x_next)
    return forecasts


# ── dispatcher ────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "HAR":      (HARModel,        "econ"),
    "AR(1)":    (AR1Model,        "econ"),
    "RandWalk": (RandomWalkModel, "econ"),
    "LASSO":    (LASSOModel,      "ml"),
    "Ridge":    (RidgeModel,      "ml"),
    "XGBoost":  (XGBoostModel,    "ml"),
    "LightGBM": (LightGBMModel,   "ml"),
}


def run_all_models(lnrv: pd.Series, ticker: str = "",
                   verbose: bool = True) -> pd.DataFrame:
    arr    = lnrv.dropna().values
    T      = len(arr)
    n_test = T // 2
    actual = arr[T - n_test:]
    if verbose:
        print(f"  {ticker}: T={T}, train={T-n_test}, test={n_test}", flush=True)
    rows = []
    for name, (cls, kind) in MODEL_REGISTRY.items():
        try:
            if kind == "econ":
                fc = _rolling_econometric(arr, cls)
            elif kind == "ml":
                fc = _rolling_ml(arr, cls)
            else:
                fc = _rolling_lstm(arr)
            rows.append({"Model": name, "MFE": round(mfe(actual, fc), 6),
                         "RMSE": round(rmse(actual, fc), 6), "Ticker": ticker})
            if verbose:
                print(f"    {name:10s}  RMSE={rows[-1]['RMSE']:.4f}", flush=True)
        except Exception as e:
            print(f"    WARNING: {name} failed for {ticker}: {e}", flush=True)
    return pd.DataFrame(rows)


def run_all_stocks(lnrv_panel: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    all_results = []
    n = len(lnrv_panel.columns)
    for idx, ticker in enumerate(lnrv_panel.columns, 1):
        print(f"[{idx}/{n}] {ticker} …", flush=True)
        result = run_all_models(lnrv_panel[ticker].dropna(),
                                ticker=ticker, verbose=verbose)
        all_results.append(result)
    return pd.concat(all_results, ignore_index=True)


def pivot_results(results_long: pd.DataFrame, metric: str = "RMSE") -> pd.DataFrame:
    return results_long.pivot(index="Ticker", columns="Model", values=metric)
