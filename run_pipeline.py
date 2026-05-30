"""
run_pipeline.py
---------------
Master script that executes the full analysis pipeline end-to-end.

Run once to generate all cached results in outputs/:
    python run_pipeline.py

The Streamlit dashboard (app.py) then loads from these cached outputs
for instant display without recomputation.

Steps:
  1. Load high-frequency data and build daily RV / lnRV panel
  2. Compute descriptive statistics (RV and lnRV)
  3. Run ADF stationarity tests
  4. Fit HAR on full sample → coefficient table
  5. Run rolling OOS forecasts for all 7 models × 30 stocks
  6. Run pairwise Granger causality tests
  7. Run VAR forecasts for top Granger pairs
  8. Save all results to outputs/
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import load_all_stocks, TICKERS
from src.rv_construction import build_rv_panel, build_lnrv_panel
from src.eda import descriptive_stats, adf_tests
from src.models import fit_har_all_stocks
from src.forecasting import run_all_stocks, pivot_results
from src.granger import run_pairwise_granger, run_var_for_significant_pairs
from src.utils import save_results, load_results, results_exist

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)


def step(n: int, name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP {n}: {name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    t0 = time.time()

    # ── STEP 1: Build RV panel ────────────────────────────────────────────────
    step(1, "Load data & build RV / lnRV panel")
    if results_exist("rv_panel.pkl") and results_exist("lnrv_panel.pkl"):
        print("  Cached. Loading from outputs/")
        rv_panel   = load_results("rv_panel.pkl")
        lnrv_panel = load_results("lnrv_panel.pkl")
    else:
        stock_data = load_all_stocks(TICKERS)
        rv_panel   = build_rv_panel(stock_data)
        lnrv_panel = build_lnrv_panel(rv_panel)
        save_results(rv_panel,   "rv_panel.pkl")
        save_results(lnrv_panel, "lnrv_panel.pkl")
        print(f"  RV panel shape: {rv_panel.shape}")

    # ── STEP 2: Descriptive statistics ────────────────────────────────────────
    step(2, "Descriptive statistics")
    if results_exist("desc_rv.pkl") and results_exist("desc_lnrv.pkl"):
        print("  Cached.")
        desc_rv   = load_results("desc_rv.pkl")
        desc_lnrv = load_results("desc_lnrv.pkl")
    else:
        rv_dict   = {t: rv_panel[t].dropna()   for t in rv_panel.columns}
        lnrv_dict = {t: lnrv_panel[t].dropna() for t in lnrv_panel.columns}
        desc_rv   = descriptive_stats(rv_dict,   label="RV")
        desc_lnrv = descriptive_stats(lnrv_dict, label="lnRV")
        save_results(desc_rv,   "desc_rv.pkl")
        save_results(desc_lnrv, "desc_lnrv.pkl")
        print(desc_rv.head())

    # ── STEP 3: ADF tests ─────────────────────────────────────────────────────
    step(3, "ADF stationarity tests on lnRV")
    if results_exist("adf_results.pkl"):
        print("  Cached.")
        adf_results = load_results("adf_results.pkl")
    else:
        lnrv_dict   = {t: lnrv_panel[t].dropna() for t in lnrv_panel.columns}
        adf_results = adf_tests(lnrv_dict)
        save_results(adf_results, "adf_results.pkl")
        print(adf_results)

    # ── STEP 4: HAR full-sample estimates ─────────────────────────────────────
    step(4, "HAR full-sample parameter estimates")
    if results_exist("har_params.pkl"):
        print("  Cached.")
        har_params = load_results("har_params.pkl")
    else:
        har_params = fit_har_all_stocks(lnrv_panel)
        save_results(har_params, "har_params.pkl")
        print(har_params)

    # ── STEP 5: Rolling OOS forecasts ─────────────────────────────────────────
    step(5, "Rolling OOS forecasts (all models × all stocks)")
    if results_exist("forecast_results.pkl"):
        print("  Cached.")
        forecast_results = load_results("forecast_results.pkl")
    else:
        forecast_results = run_all_stocks(lnrv_panel, verbose=True)
        save_results(forecast_results, "forecast_results.pkl")

    rmse_pivot = pivot_results(forecast_results, metric="RMSE")
    mfe_pivot  = pivot_results(forecast_results, metric="MFE")
    save_results(rmse_pivot, "rmse_pivot.pkl")
    save_results(mfe_pivot,  "mfe_pivot.pkl")
    print("\nAverage RMSE by model:")
    print(forecast_results.groupby("Model")["RMSE"].mean().sort_values())

    # ── STEP 6: Granger causality ─────────────────────────────────────────────
    step(6, "Pairwise Granger causality tests")
    if results_exist("granger_results.pkl"):
        print("  Cached.")
        granger_results = load_results("granger_results.pkl")
    else:
        granger_results = run_pairwise_granger(lnrv_panel, max_lag=10)
        save_results(granger_results, "granger_results.pkl")
        n_sig = granger_results["Significant"].sum()
        print(f"  Significant pairs (p<0.05): {n_sig} / {len(granger_results)}")

    # ── STEP 7: VAR forecasts for top Granger pairs ───────────────────────────
    step(7, "VAR rolling forecasts for top Granger pairs")
    if results_exist("var_results.pkl"):
        print("  Cached.")
        var_results = load_results("var_results.pkl")
    else:
        var_results = run_var_for_significant_pairs(
            lnrv_panel, granger_results, forecast_results, top_n=10
        )
        save_results(var_results, "var_results.pkl")
        print(var_results)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes.")
    print(f"  All results saved to outputs/")
    print(f"{'='*60}")
