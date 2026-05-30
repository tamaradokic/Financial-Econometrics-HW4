# Modelling and Forecasting Realized Variance — DJIA Stocks

**Financial Econometrics | ESSEC Business School | 2025–26**

---

## Overview

This project constructs and forecasts **daily Realized Variance (RV)** for all 30 constituents of the Dow Jones Industrial Average using one-minute high-frequency intraday data spanning **January 2003 to March 2022** (~4,800 trading days per stock).

RV is built from 5-minute log-returns following Andersen et al. (2001):

$$RV_t = \sum_{s=1}^{n} r_s^2, \quad r_s = \log\left(\frac{P_s}{P_{s-1}}\right)$$

Seven forecasting models are evaluated head-to-head in a rolling out-of-sample framework (last 50% of data, expanding window, h=1 day):

| Category | Model |
|---|---|
| Econometric benchmarks | HAR, AR(1), Random Walk |
| Regularised regression | LASSO, Ridge |
| Tree-based ML | XGBoost, LightGBM |
| Deep learning | LSTM |

Cross-stock predictability is assessed via **pairwise Granger causality tests** across all 870 directed pairs.

---

## Key Findings

- **HAR dominates**: The Heterogeneous Autoregressive model achieves the lowest or near-lowest RMSE across the majority of stocks, consistent with the recent literature (Kilic, 2025; Branco et al., 2024).
- **Tree-based ML is competitive**: LightGBM and XGBoost rank second on average; they match HAR on several individual stocks but offer no consistent improvement.
- **LSTM adds value selectively**: During high-volatility regimes (2008–09, 2020), LSTM's temporal memory provides modest gains; it underperforms in calm periods.
- **Granger causality is pervasive within sectors**: Significant cross-stock predictability is concentrated in financial services (JPM→GS), technology (AAPL→MSFT), and energy (CVX→DOW) — but VAR models incorporating these predictors improve OOS accuracy in only a subset of cases.
- **lnRV is stationary**: ADF tests reject the unit root for all 30 stocks at the 5% level.

---

## Project Structure

```
├── src/
│   ├── data_loader.py        Load 1-min OHLCV CSVs + timestamps
│   ├── rv_construction.py    5-min resampling → daily RV → lnRV → HAR features
│   ├── eda.py                Descriptive stats, ADF tests, Plotly charts
│   ├── models.py             HAR, AR(1), Random Walk (OLS)
│   ├── ml_models.py          LASSO, Ridge, XGBoost, LightGBM, LSTM (PyTorch)
│   ├── forecasting.py        Unified rolling OOS engine (MFE, RMSE)
│   ├── granger.py            Pairwise Granger tests + VAR forecasting
│   └── utils.py              Caching, shared plots, formatting
├── app.py                    Streamlit interactive dashboard
├── run_pipeline.py           Master script — runs all steps, saves outputs/
├── report/
│   ├── main.tex              LaTeX source (2-page academic report)
│   └── HW4_Report.pdf        Compiled PDF
├── requirements.txt
└── .gitignore
```

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add data

Place the raw data files in a `data/` folder at the project root:
- One CSV per ticker (e.g. `AAPL.csv`) — 5 columns: Open, High, Low, Close, Volume; no header
- `HFdaylistdates.csv` — shared 1-minute timestamp index

### 3. Run the pipeline

```bash
python run_pipeline.py
```

This runs all steps (RV construction → EDA → HAR → rolling forecasts → Granger tests) and caches results in `outputs/`. On a standard laptop this takes **~60–90 minutes** due to the 30-stock × 7-model rolling OOS loop. Results are cached so subsequent runs are instant.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The dashboard loads from cached outputs.

---

## Dashboard Tabs

| Tab | Content |
|---|---|
| 🏠 Overview | Project summary, key metrics |
| 📊 EDA | RV/lnRV time series, ACF, descriptive stats, cross-stock correlation |
| 🧪 Stationarity | ADF test results for all 30 stocks |
| 📐 HAR Model | Full-sample β estimates, coefficient distributions |
| 🏁 Forecasting | RMSE/MFE heatmaps, bar charts, best-model ranking |
| 🔗 Granger | Causality heatmap, top significant pairs, VAR vs HAR comparison |
| 📋 Summary | Full downloadable results tables |

---

## References

- Andersen, T.G. et al. (2001). *The Distribution of Realized Stock Return Volatility*. JFE.
- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility*. JFEC.
- Liu, L.Y., Patton, A.J., Sheppard, K. (2015). *Does Anything Beat 5-Minute RV?* JOE.
- Kilic, R. (2025). *Linear and nonlinear econometric models against ML models: RV prediction*. FEDS 2025-061.
- Branco, R.R. et al. (2024). *Forecasting realized volatility: Does anything beat linear models?* JEF.
