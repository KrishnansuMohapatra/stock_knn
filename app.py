"""
AI Stock Dashboard — main Streamlit entry-point.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd

from config import INDIA_STOCKS, GLOBAL_STOCKS, FEATURES
from styles import CUSTOM_CSS
from data_loader import load_data, load_info
from indicators import add_technical_indicators, add_ml_features
from signals import compute_signal_score, classify_signal
from ml_models import train_models, predict_next_day
from charts import (
    build_candlestick_chart,
    build_macd_chart,
    build_rsi_chart,
    build_prediction_chart,
    build_heatmap,
)
from report import generate_report

# ─────────────────────────────────────────────
# PAGE CONFIG & STYLES
# ─────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Dashboard", layout="wide", page_icon="📈")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

market = st.sidebar.radio("Market", ["🇮🇳 India", "🌍 Global", "⭐ Custom"])
if market == "🇮🇳 India":
    ticker = st.sidebar.selectbox("Select Stock", INDIA_STOCKS)
elif market == "🌍 Global":
    ticker = st.sidebar.selectbox("Select Stock", GLOBAL_STOCKS)
else:
    ticker = st.sidebar.text_input("Enter Ticker", "AAPL").upper()

start = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("today"))
k = st.sidebar.slider("KNN Neighbours", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Indicator Settings")
rsi_ob = st.sidebar.slider("RSI Overbought", 60, 90, 70)
rsi_os = st.sidebar.slider("RSI Oversold", 10, 40, 30)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", True)
show_macd = st.sidebar.checkbox("Show MACD", True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
with st.spinner("⏳ Fetching data…"):
    data = load_data(ticker, start, end)
    info = load_info(ticker)

if data.empty:
    st.error("❌ No data found. Check ticker / date range.")
    st.stop()

# ─────────────────────────────────────────────
# TITLE ROW
# ─────────────────────────────────────────────
company_name = info.get("longName", ticker)
sector = info.get("sector", "N/A")
industry = info.get("industry", "N/A")

st.title(f"📈 {company_name}")
st.caption(f"**Ticker:** {ticker} &nbsp;|&nbsp; **Sector:** {sector} &nbsp;|&nbsp; **Industry:** {industry}")

# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────
data = add_technical_indicators(data)

# ─────────────────────────────────────────────
# HELPER — latest values
# ─────────────────────────────────────────────
def last(col):
    return float(data[col].iloc[-1])

def prev(col):
    return float(data[col].iloc[-2])

price = last("Close")
prev_price = prev("Close")
day_chg = price - prev_price
day_pct = (day_chg / prev_price) * 100

rsi_val = last("RSI")
macd_val = last("MACD")
macd_sig = last("MACD_S")
macd_hist = last("MACD_H")
atr_val = last("ATR")
adx_val = last("ADX")
cci_val = last("CCI")
willr_val = last("WILLR")
obv_val = last("OBV")
stoch_k = last("STOCH_K")
stoch_d = last("STOCH_D")
bb_u = last("BB_U")
bb_l = last("BB_L")
bb_m = last("BB_M")

# ─────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("💰 Price",
          f"₹{price:,.2f}" if ".NS" in ticker else f"${price:,.2f}",
          f"{day_chg:+.2f} ({day_pct:+.2f}%)")
c2.metric("📈 52W High", f"{float(data['High'].tail(252).max()):,.2f}")
c3.metric("📉 52W Low", f"{float(data['Low'].tail(252).min()):,.2f}")
c4.metric("📊 RSI (14)", f"{rsi_val:.1f}")
c5.metric("⚡ ATR", f"{atr_val:.2f}")
c6.metric("🔊 Volume", f"{int(data['Volume'].iloc[-1]):,}")
st.markdown("---")

# ─────────────────────────────────────────────
# SIGNAL ENGINE
# ─────────────────────────────────────────────
signals, score = compute_signal_score(data, rsi_ob, rsi_os)
signal_label, signal_cls, signal_emoji = classify_signal(score)

st.subheader("🤖 AI Trading Signal")
left, right = st.columns([1, 2])

with left:
    st.markdown(f"""
    <div class="signal-card {signal_cls}">
        {signal_emoji} {signal_label}<br>
        <span style="font-size:14px;opacity:.8">Composite Score: {score:+d} / 100</span>
    </div>
    """, unsafe_allow_html=True)

    bar_color = ("#52b788" if score > 0 else "#f87171") if score != 0 else "#fbbf24"
    bar_pct = abs(score)
    st.markdown(f"""
    <div style="background:#2d3748;border-radius:6px;height:14px;width:100%">
      <div style="background:{bar_color};width:{bar_pct}%;height:14px;
                  border-radius:6px;transition:width .5s"></div>
    </div>
    <p style="color:#9ca3af;font-size:12px;margin-top:4px">
      Signal strength: {bar_pct}%
    </p>
    """, unsafe_allow_html=True)

with right:
    st.markdown("**📋 Indicator Breakdown**")
    for name, (label, tone, pts) in signals.items():
        arrow = "▲" if pts > 0 else ("▼" if pts < 0 else "─")
        st.markdown(f"""
        <div class="indicator-row">
          <span class="metric-label">{name}</span>
          <span class="{tone}">{arrow} {label}</span>
          <span class="metric-value">{pts:+d}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
st.subheader("📊 Interactive Price Chart")
st.plotly_chart(build_candlestick_chart(data, show_bb), use_container_width=True)

if show_macd:
    col_macd, col_rsi = st.columns(2)
    with col_macd:
        st.markdown("**MACD**")
        st.plotly_chart(build_macd_chart(data), use_container_width=True)
    with col_rsi:
        st.markdown("**RSI (14)**")
        st.plotly_chart(build_rsi_chart(data, rsi_ob, rsi_os), use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# ML MODELS
# ─────────────────────────────────────────────
st.subheader("🧠 Machine Learning Price Prediction")

data = add_ml_features(data)
results = train_models(data, FEATURES, k)

st.markdown("**Model Comparison**")
st.dataframe(
    results["perf_df"].style.highlight_min(axis=0, color="#1a472a"),
    use_container_width=True,
)

st.plotly_chart(
    build_prediction_chart(results["y_test"], results["predictions"]),
    use_container_width=True,
)

preds, pred_avg, cur_sym = predict_next_day(results, price, ticker)

st.markdown("**🔮 Next-Day Price Forecast**")
cols = st.columns(5)
model_names = list(preds.keys())
for i, name in enumerate(model_names):
    p = preds[name]
    cols[i].metric(name, f"{cur_sym}{p:.2f}", f"{((p - price) / price) * 100:+.2f}%")
cols[4].metric("🎯 Ensemble Avg", f"{cur_sym}{pred_avg:.2f}",
               f"{((pred_avg - price) / price) * 100:+.2f}%")

st.markdown("---")

# ─────────────────────────────────────────────
# AUTO ANALYSIS REPORT
# ─────────────────────────────────────────────
st.subheader("📝 Auto Analysis Report")

info_vals = dict(
    price=price, day_chg=day_chg, day_pct=day_pct,
    rsi_val=rsi_val, macd_val=macd_val, macd_sig=macd_sig,
    macd_hist=macd_hist, stoch_k=stoch_k, stoch_d=stoch_d,
    adx_val=adx_val, cci_val=cci_val, willr_val=willr_val,
    atr_val=atr_val, obv_val=obv_val,
    bb_u=bb_u, bb_l=bb_l, bb_m=bb_m,
    rsi_ob=rsi_ob, rsi_os=rsi_os,
)
signal_info = dict(score=score, signal_label=signal_label)
ml_info = dict(
    perf_df=results["perf_df"], pred_avg=pred_avg,
    X_train_len=len(results["X_train"]), X_test_len=len(results["X_test"]),
)

st.markdown(
    generate_report(data, ticker, company_name, info_vals, signal_info, ml_info),
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────
# MARKET HEATMAP
# ─────────────────────────────────────────────
st.subheader("🌍 Market Heatmap (5-Day Return)")
with st.spinner("Loading heatmap…"):
    st.plotly_chart(build_heatmap(), use_container_width=True)

# ─────────────────────────────────────────────
# RAW DATA
# ─────────────────────────────────────────────
with st.expander("📄 View Raw Data"):
    st.dataframe(
        data[["Open", "High", "Low", "Close", "Volume",
              "MA20", "MA50", "RSI", "MACD", "ATR"]].tail(30).round(2),
        use_container_width=True,
    )

st.markdown("""
<hr>
<p style="text-align:center;color:#6b7280;font-size:13px">
  Built with ❤️ using Streamlit · yFinance · scikit-learn · Plotly · TA-Lib
</p>
""", unsafe_allow_html=True)