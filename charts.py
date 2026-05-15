"""
Plotly chart builders for the Streamlit dashboard.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

from config import ALL_TICKERS


def build_candlestick_chart(data, show_bb):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"],
        name="Price",
        increasing_line_color="#52b788",
        decreasing_line_color="#f87171",
    ))
    for col, color, dash in [("MA20", "#60a5fa", "solid"),
                              ("MA50", "#fbbf24", "dot"),
                              ("MA200", "#f87171", "dash")]:
        fig.add_trace(go.Scatter(
            x=data.index, y=data[col],
            name=col, line=dict(color=color, dash=dash, width=1.5),
        ))
    if show_bb:
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


def build_macd_chart(data):
    fm = go.Figure()
    fm.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color="#60a5fa")))
    fm.add_trace(go.Scatter(x=data.index, y=data["MACD_S"], name="Signal", line=dict(color="#fbbf24")))
    colors = ["#52b788" if v >= 0 else "#f87171" for v in data["MACD_H"]]
    fm.add_trace(go.Bar(x=data.index, y=data["MACD_H"], name="Histogram", marker_color=colors))
    fm.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=10, b=10))
    return fm


def build_rsi_chart(data, rsi_ob, rsi_os):
    fr = go.Figure()
    fr.add_trace(go.Scatter(
        x=data.index, y=data["RSI"], name="RSI",
        line=dict(color="#a78bfa"), fill="tozeroy",
        fillcolor="rgba(167,139,250,0.1)",
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


def build_prediction_chart(y_test, predictions):
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=y_test.values, name="Actual", line=dict(color="#f3f4f6", width=2)))
    styles = {
        "KNN": ("#60a5fa", "dot"), "Ridge": ("#fbbf24", "dot"),
        "Random Forest": ("#52b788", "dot"), "Gradient Boosting": ("#a78bfa", "dot"),
    }
    for name, pred in predictions.items():
        color, dash = styles.get(name, ("#ffffff", "dot"))
        fig4.add_trace(go.Scatter(y=pred, name=name, line=dict(color=color, dash=dash)))
    fig4.update_layout(
        template="plotly_dark", height=380,
        title="Actual vs Predicted (Test Set)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig4


def build_heatmap():
    heat = []
    for t, name in ALL_TICKERS.items():
        try:
            df_h = yf.download(t, period="5d", progress=False)
            if isinstance(df_h.columns, pd.MultiIndex):
                df_h.columns = df_h.columns.get_level_values(0)
            if len(df_h) >= 2:
                chg = ((float(df_h["Close"].iloc[-1]) - float(df_h["Close"].iloc[0])) /
                       float(df_h["Close"].iloc[0])) * 100
                heat.append({"Stock": name, "Ticker": t, "Change": round(chg, 2)})
        except:
            pass

    heat_df = pd.DataFrame(heat)
    heat_df["Abs"] = heat_df["Change"].abs()

    fig_h = px.treemap(
        heat_df, path=["Stock"], values="Abs",
        color="Change", color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        hover_data={"Change": ":.2f", "Ticker": True},
    )
    fig_h.update_traces(textinfo="label+value", texttemplate="<b>%{label}</b><br>%{color:.2f}%")
    fig_h.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=10, b=10))
    return fig_h
