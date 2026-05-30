# Modelling and Forecasting Realized Variance — DJIA Stocks

**Financial Econometrics | ESSEC Business School | 2025–26**

---

## Overview

This project constructs and forecasts **daily Realized Variance (RV)** for all 30 constituents of the Dow Jones Industrial Average using one-minute high-frequency intraday data spanning **January 2003 to March 2022** (~4,800 trading days per stock).

RV is built from 5-minute log-returns following Andersen et al. (2001):

$$RV_t = \sum_{s=1}^{n} r_s^2, \quad r_s = \log\left(\frac{P_s}{P_{s-1}}\right)$$

Seven forecasting models are evaluated head-to-head in a rolling out-of-sample framework (last 50% of data, annual retraining, h=1 day):

| Category | Model |
|---|---|
| Econometric benchmarks | HAR, AR(1), Random Walk |
| Regularised regression | LASSO, Ridge |
| Tree-based ML | XGBoost, LightGBM |

Cross-stock predictability is assessed via **pairwise Granger causality tests** across all 870 directed pairs, with bivariate VAR models used to test whether detected causality improves forecasting (Q12).

---

## Key Findings

- **HAR, LASSO, and Ridge** achieve near-identical best RMSE (≈0.511), confirming the difficulty of beating a well-specified linear benchmark (Kilic, 2025).
- **XGBoost and LightGBM** are competitive but offer no consistent advantage over HAR.
- **Random Walk** is always worst, confirming significant predictability in lnRV.
- **869 out of 870 Granger pairs** are significant at 5% — volatility spillovers are pervasive across the DJIA.
- **VAR models** improve on HAR in only 4 out of 10 tested pairs, with modest RMSE gains (~0.004), suggesting Granger causality is statistically present but has limited forecasting value.
- **lnRV is stationary** for all 30 stocks (ADF tests reject unit root at 5%).

---

## Project Structure

```
├── src/
│   ├── data_loader.py        Load 1-min OHLCV CSVs + timestamps
│   ├── rv_construction.py    5-min resampling → daily RV → lnRV → HAR features
│   ├── eda.py                Descriptive stats, ADF tests, Plotly charts
│   ├── models.py             HAR, AR(1), Random Walk (OLS)
│   ├── ml_models.py          LASSO, Ridge, XGBoost, LightGBM
│   ├── forecasting.py        Unified rolling OOS engine (MFE, RMSE)
│   ├── granger.py            Pairwise Granger tests + VAR forecasting
│   └── utils.py              Caching, shared plots, formatting
├── outputs/                  Pre-computed results (loaded by dashboard)
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

### 3. Run the pipeline (optional — outputs already included)

```bash
python run_pipeline.py
```

Results are pre-computed and saved in `outputs/`. This step is only needed if you want to re-run the analysis from scratch.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

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

## Reference

Kilic, R. (2025). *Linear and nonlinear econometric models against machine learning models: realized volatility prediction.* Finance and Economics Discussion Series, 2025-061.
