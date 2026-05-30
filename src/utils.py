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
    """Pickle an object to the outputs/ directory."""
    path = OUTPUTS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")


def load_results(filename: str):
    """Load a pickled object from outputs/."""
    path = OUTPUTS_DIR / filename
    with open(path, "rb") as f:
        return pickle.load(f)


def results_exist(filename: str) -> bool:
    return (OUTPUTS_DIR / filename).exists()


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
