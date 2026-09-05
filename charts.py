"""
Plotly chart builders for the Streamlit dashboard.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_candlestick_chart(data: pd.DataFrame, show_bb: bool):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"],
        name="Price",
        increasing_line_color="#52b788",
        decreasing_line_color="#f87171",
    ))
    for col, color, dash in [
        ("MA20", "#60a5fa", "solid"),
        ("MA50", "#fbbf24", "dot"),
        ("MA200", "#f87171", "dash"),
    ]:
        if col in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col],
                name=col, line=dict(color=color, dash=dash, width=1.5),
            ))
    if show_bb and "BB_U" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["BB_U"],
            name="BB Upper", line=dict(color="#a78bfa", width=1, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data["BB_L"],
            name="BB Lower", line=dict(color="#a78bfa", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(167,139,250,0.05)",
        ))
    fig.update_layout(
        template="plotly_dark", xaxis_rangeslider_visible=False,
        height=520, legend=dict(orientation="h", y=1.02),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def build_macd_chart(data: pd.DataFrame):
    fm = go.Figure()
    fm.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color="#60a5fa")))
    fm.add_trace(go.Scatter(x=data.index, y=data["MACD_S"], name="Signal", line=dict(color="#fbbf24")))
    colors = ["#52b788" if v >= 0 else "#f87171" for v in data["MACD_H"].fillna(0)]
    fm.add_trace(go.Bar(x=data.index, y=data["MACD_H"], name="Histogram", marker_color=colors))
    fm.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=10, b=10))
    return fm


def build_rsi_chart(data: pd.DataFrame, rsi_ob: int, rsi_os: int):
    fr = go.Figure()
    fr.add_trace(go.Scatter(
        x=data.index, y=data["RSI"], name="RSI",
        line=dict(color="#a78bfa"),
    ))
    fr.add_hline(y=rsi_ob, line_dash="dash", line_color="#f87171", annotation_text=f"OB {rsi_ob}")
    fr.add_hline(y=rsi_os, line_dash="dash", line_color="#52b788", annotation_text=f"OS {rsi_os}")
    fr.add_hline(y=50, line_dash="dot", line_color="#6b7280")
    fr.update_layout(
        template="plotly_dark", height=280,
        yaxis=dict(range=[0, 100]),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fr


def build_prediction_chart(y_test: pd.Series, predictions: dict):
    fig = go.Figure()
    x = y_test.index
    fig.add_trace(go.Scatter(x=x, y=y_test.values, name="Actual next close", line=dict(color="#f3f4f6", width=2)))
    styles = {
        "KNN": ("#60a5fa", "dot"),
        "Ridge": ("#fbbf24", "dot"),
        "Random Forest": ("#52b788", "dot"),
        "Gradient Boosting": ("#a78bfa", "dot"),
        "Persistence": ("#9ca3af", "dash"),
    }
    for name, pred in predictions.items():
        color, dash = styles.get(name, ("#ffffff", "dot"))
        width = 2 if name == "Persistence" else 1.5
        fig.add_trace(go.Scatter(
            x=x, y=pred, name=name,
            line=dict(color=color, dash=dash, width=width),
        ))
    fig.update_layout(
        template="plotly_dark", height=380,
        title="Next-close forecast vs actual (chronological test window)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def build_equity_chart(equity: pd.Series, buy_hold: pd.Series):
    fig = go.Figure()
    if equity is not None and len(equity):
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values, name="Signal strategy",
            line=dict(color="#60a5fa", width=2),
        ))
    if buy_hold is not None and len(buy_hold):
        fig.add_trace(go.Scatter(
            x=buy_hold.index, y=buy_hold.values, name="Buy & hold",
            line=dict(color="#9ca3af", width=1.5, dash="dash"),
        ))
    fig.update_layout(
        template="plotly_dark", height=340,
        title="Paper equity vs buy & hold (next-open fills, costs included)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def build_heatmap(heat_df: pd.DataFrame | None, names: dict | None = None):
    names = names or {}
    if heat_df is None or heat_df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", height=360,
            title="No heatmap data (Yahoo rate-limit, network, or empty universe)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    df = heat_df.copy()
    df["Stock"] = df["Ticker"].map(lambda t: names.get(t, t))
    df["Abs"] = df["Change"].abs().clip(lower=0.01)

    fig_h = px.treemap(
        df, path=["Stock"], values="Abs",
        color="Change", color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        hover_data={"Change": ":.2f", "Ticker": True},
    )
    fig_h.update_traces(textinfo="label+value", texttemplate="<b>%{label}</b><br>%{color:.2f}%")
    fig_h.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=10, b=10))
    return fig_h
