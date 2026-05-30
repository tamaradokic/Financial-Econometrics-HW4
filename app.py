"""
app.py
------
Interactive Streamlit dashboard for Homework 4: Modelling & Forecasting
Realized Variance on Dow Jones Index constituents.

Run with:
    streamlit run app.py

Requires outputs/ to be populated first:
    python run_pipeline.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from src.utils import load_results, results_exist, plot_rmse_bar, plot_rmse_heatmap
from src.eda import (
    plot_rv_timeseries, plot_lnrv_timeseries, plot_acf,
    plot_rv_panel_heatmap,
)
from src.granger import granger_heatmap, top_granger_pairs
from src.rv_construction import build_lnrv_panel

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DJIA Realized Variance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUTS = Path("outputs")


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_all_outputs():
    """Load all cached pipeline outputs once and cache in session."""
    out = {}
    files = {
        "rv_panel":         "rv_panel.pkl",
        "lnrv_panel":       "lnrv_panel.pkl",
        "desc_rv":          "desc_rv.pkl",
        "desc_lnrv":        "desc_lnrv.pkl",
        "adf_results":      "adf_results.pkl",
        "har_params":       "har_params.pkl",
        "forecast_results": "forecast_results.pkl",
        "rmse_pivot":       "rmse_pivot.pkl",
        "mfe_pivot":        "mfe_pivot.pkl",
        "granger_results":  "granger_results.pkl",
        "var_results":      "var_results.pkl",
    }
    missing = []
    for key, fname in files.items():
        if results_exist(fname):
            out[key] = load_results(fname)
        else:
            missing.append(fname)
    return out, missing


def section(title: str) -> None:
    st.markdown(f"### {title}")


# ── sidebar ───────────────────────────────────────────────────────────────────

def sidebar(data: dict) -> str:
    st.sidebar.title("📈 DJIA Realized Variance")
    st.sidebar.markdown("**Financial Econometrics HW4**")
    st.sidebar.markdown("---")

    tickers = sorted(data["rv_panel"].columns.tolist())
    ticker  = st.sidebar.selectbox("Select stock", tickers, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Data:** Jan 2003 – Mar 2022  \n"
        "**Stocks:** 30 DJIA constituents  \n"
        "**RV frequency:** Daily (5-min sampling)  \n"
        "**Models:** HAR, AR(1), RW, LASSO, Ridge, XGBoost, LightGBM, LSTM"
    )
    return ticker


# ── tabs ──────────────────────────────────────────────────────────────────────

def tab_overview(data: dict) -> None:
    st.markdown(
        """
        ## Modelling & Forecasting Realized Variance — DJIA Stocks

        This dashboard presents a full econometric and machine learning analysis
        of **daily Realized Variance (RV)** for the 30 constituents of the
        Dow Jones Industrial Average, using high-frequency intraday data
        from **January 2003 to March 2022**.

        **Realized Variance** is constructed as the sum of squared 5-minute
        log-returns within each trading day, following [Andersen et al. (2001)](https://doi.org/10.1016/S0304-405X(01)00055-1):

        $$RV_t = \\sum_{s=1}^{n} r_s^2, \\quad r_s = \\log\\left(\\frac{P_s}{P_{s-1}}\\right)$$

        **Navigate the tabs** to explore:
        | Tab | Content |
        |-----|---------|
        | 📊 EDA | Time series, ACF plots, descriptive stats |
        | 🧪 Stationarity | ADF test results for all stocks |
        | 📐 HAR Model | Full-sample parameter estimates |
        | 🏁 Forecasting | OOS comparison: econometric vs ML models |
        | 🔗 Granger | Cross-stock predictability heatmap |
        | 📋 Summary | Full results tables |
        """
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stocks analysed", len(data["rv_panel"].columns))
    with col2:
        avg_T = int(data["lnrv_panel"].notna().sum().mean())
        st.metric("Avg. trading days per stock", avg_T)
    with col3:
        n_sig = data["granger_results"]["Significant"].sum()
        st.metric("Significant Granger pairs (p<5%)", n_sig)


def tab_eda(data: dict, ticker: str) -> None:
    rv_series   = data["rv_panel"][ticker].dropna()
    lnrv_series = data["lnrv_panel"][ticker].dropna()

    section("Realized Variance & log-RV Time Series")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_rv_timeseries(rv_series, ticker),
                        use_container_width=True)
    with col2:
        st.plotly_chart(plot_lnrv_timeseries(lnrv_series, ticker),
                        use_container_width=True)

    section("Sample Autocorrelation Function (lnRV)")
    st.info("The slow ACF decay confirms the well-known **long-memory** property "
            "of realized variance — motivation for the HAR model.")
    st.plotly_chart(plot_acf(lnrv_series, ticker, n_lags=60),
                    use_container_width=True)

    section("Descriptive Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("**RV** (raw realized variance)")
        row_rv = data["desc_rv"].loc[[ticker]] if ticker in data["desc_rv"].index else pd.DataFrame()
        st.dataframe(row_rv, use_container_width=True)
    with col2:
        st.caption("**lnRV** (log-transformed)")
        row_ln = data["desc_lnrv"].loc[[ticker]] if ticker in data["desc_lnrv"].index else pd.DataFrame()
        st.dataframe(row_ln, use_container_width=True)

    section("Cross-stock lnRV Correlation")
    st.plotly_chart(plot_rv_panel_heatmap(data["lnrv_panel"]),
                    use_container_width=True)


def tab_stationarity(data: dict, ticker: str) -> None:
    section("Augmented Dickey-Fuller Tests on lnRV")
    st.markdown(
        "H₀: unit root (non-stationary) | H₁: stationary  \n"
        "Lag length selected by AIC. Rejection at 5% significance level."
    )
    adf = data["adf_results"]
    # Highlight the selected ticker
    def highlight_selected(row):
        return ["background-color: #dbeafe"] * len(row) \
               if row.name == ticker else [""] * len(row)

    st.dataframe(
        adf.style.apply(highlight_selected, axis=1).format({
            "ADF Stat": "{:.4f}", "p-value": "{:.4f}"
        }),
        use_container_width=True, height=600,
    )

    n_stationary = (adf["Stationary (5%)"] == "Yes").sum()
    st.success(f"**{n_stationary} / {len(adf)} stocks** reject the unit root at 5% → lnRV is stationary.")


def tab_har(data: dict, ticker: str) -> None:
    section("HAR Model — Full-Sample Parameter Estimates")
    st.markdown(
        r"""
        The **Heterogeneous Autoregressive (HAR)** model (Corsi, 2009) decomposes
        realized variance into daily, weekly, and monthly components:

        $$y_t = \beta_0 + \beta_d \cdot y_{t-1} + \beta_w \cdot \bar{y}^{(5)}_{t-1}
               + \beta_m \cdot \bar{y}^{(22)}_{t-1} + \varepsilon_t$$

        This is a **restricted AR(22)** estimated by OLS, capturing the heterogeneity
        of market participants operating at different horizons.
        """
    )

    params = data["har_params"]

    # Highlight selected ticker
    def highlight(row):
        return ["background-color: #dbeafe"] * len(row) \
               if row.name == ticker else [""] * len(row)

    st.dataframe(
        params.style.apply(highlight, axis=1).format("{:.4f}"),
        use_container_width=True, height=600,
    )

    section("Coefficient Distributions Across Stocks")
    fig = go.Figure()
    for col, color in zip(["β_d", "β_w", "β_m"],
                           ["#2563EB", "#7C3AED", "#059669"]):
        fig.add_trace(go.Box(y=params[col], name=col, marker_color=color,
                             boxmean=True))
    fig.update_layout(template="plotly_white", height=350,
                      yaxis_title="Coefficient value",
                      title="Distribution of HAR coefficients across 30 stocks")
    st.plotly_chart(fig, use_container_width=True)


def tab_forecasting(data: dict, ticker: str) -> None:
    section("Out-of-Sample Forecasting — All Models")
    st.markdown(
        "**Setup:** Last 50% of observations used for testing. "
        "Models re-estimated after every single forecast (expanding window). "
        "Horizon: h = 1 day."
    )

    results = data["forecast_results"]
    rmse_piv = data["rmse_pivot"]
    mfe_piv  = data["mfe_pivot"]

    col1, col2 = st.columns(2)
    with col1:
        st.caption("📊 RMSE for selected stock")
        st.plotly_chart(plot_rmse_bar(results, ticker=ticker),
                        use_container_width=True)
    with col2:
        st.caption("📊 Average RMSE across all stocks")
        avg_rmse = results.groupby("Model")["RMSE"].mean().reset_index()
        st.plotly_chart(
            px.bar(avg_rmse, x="Model", y="RMSE", text_auto=".4f",
                   color="Model", template="plotly_white",
                   title="Mean RMSE — all stocks",
                   color_discrete_sequence=px.colors.qualitative.Plotly),
            use_container_width=True,
        )

    section("RMSE Heatmap (all stocks × all models)")
    st.plotly_chart(plot_rmse_heatmap(rmse_piv), use_container_width=True)

    section("MFE Heatmap (bias)")
    st.caption("Positive MFE = systematic over-forecasting; negative = under-forecasting")
    from src.utils import plot_rmse_heatmap as hm
    fig_mfe = go.Figure(go.Heatmap(
        z=mfe_piv.values,
        x=mfe_piv.columns.tolist(),
        y=mfe_piv.index.tolist(),
        colorscale="RdBu", zmid=0,
        text=np.round(mfe_piv.values, 4),
        texttemplate="%{text}",
        textfont={"size": 8},
        colorbar=dict(title="MFE"),
    ))
    fig_mfe.update_layout(
        title="MFE Heatmap",
        height=700, width=800,
        template="plotly_white",
        margin=dict(l=80, r=20, t=60, b=80),
    )
    st.plotly_chart(fig_mfe, use_container_width=True)

    section("Best Model per Stock")
    best = rmse_piv.idxmin(axis=1).reset_index()
    best.columns = ["Ticker", "Best Model"]
    best["RMSE"] = rmse_piv.min(axis=1).values.round(4)
    counts = best["Best Model"].value_counts().reset_index()
    counts.columns = ["Model", "# Stocks where Best"]
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(best, use_container_width=True, height=500)
    with col2:
        st.dataframe(counts, use_container_width=True)


def tab_granger(data: dict) -> None:
    section("Pairwise Granger Causality — All DJIA Stocks")
    st.markdown(
        "**Granger causality:** past values of stock X help predict stock Y "
        "beyond Y's own history.  \n"
        "Tested using the minimum F-test p-value across lags 1–10."
    )

    granger = data["granger_results"]
    tickers = sorted(set(granger["From"].tolist() + granger["To"].tolist()))

    st.plotly_chart(granger_heatmap(granger, tickers), use_container_width=True)

    section("Top 20 Most Significant Pairs")
    top = top_granger_pairs(granger, n=20)
    st.dataframe(top.style.format({"p_value": "{:.4f}"}),
                 use_container_width=True)

    section("VAR vs HAR — Do Granger Pairs Improve Forecasting?")
    st.markdown(
        "For the top-10 Granger pairs, a **bivariate VAR** model is fitted "
        "and compared against the univariate HAR. Positive improvement = VAR wins."
    )
    var_res = data["var_results"]
    st.dataframe(
        var_res.style.format({
            "VAR RMSE": "{:.4f}",
            "HAR RMSE (Y)": "{:.4f}",
            "Improvement": "{:.4f}",
            "p-value": "{:.4f}",
        }).applymap(
            lambda v: "background-color: #bbf7d0" if v is True else
                      ("background-color: #fecaca" if v is False else ""),
            subset=["Better?"],
        ),
        use_container_width=True,
    )


def tab_summary(data: dict) -> None:
    section("Full Summary Tables")

    tab_a, tab_b, tab_c, tab_d = st.tabs([
        "Descriptive Stats (RV)", "Descriptive Stats (lnRV)",
        "RMSE — All Models", "MFE — All Models",
    ])

    with tab_a:
        st.dataframe(data["desc_rv"].style.format("{:.4f}"),
                     use_container_width=True)
    with tab_b:
        st.dataframe(data["desc_lnrv"].style.format("{:.4f}"),
                     use_container_width=True)
    with tab_c:
        st.dataframe(data["rmse_pivot"].style.format("{:.4f}")
                     .highlight_min(axis=1, color="#bbf7d0"),
                     use_container_width=True)
    with tab_d:
        st.dataframe(data["mfe_pivot"].style.format("{:.4f}"),
                     use_container_width=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    data, missing = load_all_outputs()

    if missing:
        st.error(
            f"Some pipeline outputs are missing: {missing}\n\n"
            "Please run `python run_pipeline.py` first to generate all results."
        )
        st.stop()

    ticker = sidebar(data)

    tabs = st.tabs([
        "🏠 Overview",
        "📊 EDA",
        "🧪 Stationarity",
        "📐 HAR Model",
        "🏁 Forecasting",
        "🔗 Granger",
        "📋 Summary",
    ])

    with tabs[0]: tab_overview(data)
    with tabs[1]: tab_eda(data, ticker)
    with tabs[2]: tab_stationarity(data, ticker)
    with tabs[3]: tab_har(data, ticker)
    with tabs[4]: tab_forecasting(data, ticker)
    with tabs[5]: tab_granger(data)
    with tabs[6]: tab_summary(data)


if __name__ == "__main__":
    main()
