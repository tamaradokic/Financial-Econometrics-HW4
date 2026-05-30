"""
utils.py
--------
Shared utility functions used across the project.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pickle
import os

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ── persistence ───────────────────────────────────────────────────────────────

def save_results(obj, filename: str) -> None:
    """Save a DataFrame to outputs/ as parquet (portable) and pickle (fast local)."""
    import pandas as pd
    path_pkl = OUTPUTS_DIR / filename
    with open(path_pkl, "wb") as f:
        pickle.dump(obj, f)
    # Also save as parquet for cross-platform use (Streamlit Cloud)
    if isinstance(obj, pd.DataFrame):
        pq_name = filename.replace(".pkl", ".parquet")
        obj.to_parquet(OUTPUTS_DIR / pq_name)
    print(f"  Saved → {path_pkl}")


def load_results(filename: str):
    """Load from parquet if available, else pickle."""
    import pandas as pd
    pq_path  = OUTPUTS_DIR / filename.replace(".pkl", ".parquet")
    pkl_path = OUTPUTS_DIR / filename
    if pq_path.exists():
        return pd.read_parquet(pq_path)
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def results_exist(filename: str) -> bool:
    pq  = OUTPUTS_DIR / filename.replace(".pkl", ".parquet")
    pkl = OUTPUTS_DIR / filename
    return pq.exists() or pkl.exists()


# ── forecast comparison plots ─────────────────────────────────────────────────

def plot_forecast_comparison(
    actual: np.ndarray,
    forecasts_dict: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    ticker: str,
) -> go.Figure:
    """
    Overlay actual vs. predicted lnRV for multiple models on one chart.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        mode="lines", name="Actual",
        line=dict(color="black", width=1.5),
    ))
    colors = px.colors.qualitative.Plotly
    for i, (model_name, fc) in enumerate(forecasts_dict.items()):
        fig.add_trace(go.Scatter(
            x=dates, y=fc,
            mode="lines", name=model_name,
            line=dict(width=1, dash="dot", color=colors[i % len(colors)]),
        ))
    fig.update_layout(
        title=f"{ticker} – OOS Forecast vs Actual lnRV",
        xaxis_title="Date", yaxis_title="lnRV",
        template="plotly_white", height=400,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=50, r=20, t=50, b=60),
    )
    return fig


def plot_rmse_bar(results_df: pd.DataFrame,
                  ticker: str = "all") -> go.Figure:
    """
    Bar chart of RMSE by model, averaged across stocks (or for one ticker).
    """
    if ticker != "all":
        df = results_df[results_df["Ticker"] == ticker]
        title = f"{ticker} – RMSE by Model"
    else:
        df = results_df.groupby("Model")["RMSE"].mean().reset_index()
        title = "Average RMSE by Model (all stocks)"

    fig = px.bar(
        df, x="Model", y="RMSE",
        color="Model", text_auto=".4f",
        title=title,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_layout(height=400, showlegend=False,
                      margin=dict(l=50, r=20, t=50, b=60))
    fig.update_traces(textposition="outside")
    return fig


def plot_rmse_heatmap(pivot_rmse: pd.DataFrame) -> go.Figure:
    """
    Heatmap of RMSE: rows = tickers, columns = models.
    Cells with lower RMSE are darker green.
    """
    fig = go.Figure(go.Heatmap(
        z=pivot_rmse.values,
        x=pivot_rmse.columns.tolist(),
        y=pivot_rmse.index.tolist(),
        colorscale="YlOrRd",
        text=np.round(pivot_rmse.values, 4),
        texttemplate="%{text}",
        textfont={"size": 8},
        colorbar=dict(title="RMSE"),
    ))
    fig.update_layout(
        title="RMSE Heatmap (rows = tickers, cols = models)",
        height=700, width=800,
        template="plotly_white",
        margin=dict(l=80, r=20, t=60, b=80),
    )
    return fig


def style_table(df: pd.DataFrame) -> pd.DataFrame:
    """Round all numeric columns to 4 decimal places for display."""
    return df.round(4)
