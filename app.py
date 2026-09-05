"""
AI Stock Dashboard — Streamlit entry-point.

Paper blotter + next-open backtest.
India: NSE official. Global: Stooq. Yahoo only as last resort.
Not a live broker.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from backtest import run_signal_backtest
from broker import empty_book, equity_by_ccy, execute, suggest_trade
from charts import (
    build_candlestick_chart,
    build_equity_chart,
    build_heatmap,
    build_macd_chart,
    build_prediction_chart,
    build_rsi_chart,
)
from config import (
    ALL_TICKERS,
    DEFAULT_CASH_INR,
    DEFAULT_CASH_USD,
    DEFAULT_RISK_PCT,
    FEATURES,
    GLOBAL_STOCKS,
    INDIA_STOCKS,
    MIN_BARS,
)
from data_loader import load_heatmap_returns, load_market, make_demo_ohlcv
from indicators import add_ml_features, add_technical_indicators
from ml_models import predict_next_day, train_models
from report import generate_report
from signals import classify_signal, compute_signal_score
from styles import CUSTOM_CSS
from utils import (
    currency_for,
    esc,
    fmt_money,
    fmt_volume,
    last_valid,
    money_symbol,
    safe_float,
)

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
    raw = st.sidebar.text_input("Enter Ticker", "AAPL")
    ticker = re.sub(r"[^A-Z0-9._\-=^]", "", (raw or "").upper())

start = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("today"))
k = st.sidebar.slider("KNN Neighbours", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Indicator Settings")
rsi_ob = st.sidebar.slider("RSI Overbought", 60, 90, 70)
rsi_os = st.sidebar.slider("RSI Oversold", 10, 40, 30)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", True)
show_macd = st.sidebar.checkbox("Show MACD", True)
show_rsi = st.sidebar.checkbox("Show RSI", True)

use_demo = st.sidebar.checkbox("Offline demo data (no Yahoo)", False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏦 Paper broker (simulated)")
risk_pct = st.sidebar.slider("Risk per trade", 0.25, 5.0, DEFAULT_RISK_PCT * 100, 0.25) / 100.0
long_short = st.sidebar.checkbox("Backtest allows shorts", False)
if st.sidebar.button("Reset paper book"):
    st.session_state.paper_book = empty_book()
    st.rerun()

if "paper_book" not in st.session_state:
    st.session_state.paper_book = empty_book()

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
if not ticker:
    st.error("Enter a valid ticker (letters, digits, `. - = ^`). India: RELIANCE.NS")
    st.stop()

if start > end:
    st.error("Start date must be on or before end date.")
    st.stop()

demo_mode = bool(use_demo)
quote = {}
source_id = "demo" if demo_mode else "none"
source_label = "built-in demo OHLCV (not market data)"
source_note = ""
live = False

if use_demo:
    data = make_demo_ohlcv(start, end)
    info = {"longName": f"Demo series ({ticker})", "sector": "Demo", "industry": "Offline simulation"}
else:
    with st.spinner("⏳ Fetching exchange data (NSE → Stooq → Yahoo fallback)…"):
        market = load_market(ticker, start, end)
    data = market.get("ohlcv", pd.DataFrame())
    info = market.get("info") or {}
    quote = market.get("quote") or {}
    source_id = market.get("source_id") or "none"
    source_label = market.get("source") or "none"
    source_note = market.get("note") or ""
    live = bool(market.get("live"))
    if data.empty:
        st.warning(
            "NSE / Stooq / Yahoo all returned no bars (network, SSL, or bad ticker). "
            "Loading built-in demo data so the paper broker still works."
        )
        data = make_demo_ohlcv(start, end)
        info = {"longName": f"Demo series ({ticker})", "sector": "Demo", "industry": "Offline simulation"}
        demo_mode = True
        source_id = "demo"
        source_label = "built-in demo OHLCV (not market data)"
        live = False

if len(data) < MIN_BARS:
    st.error(
        f"❌ Need at least {MIN_BARS} trading days for indicators. "
        f"Got {len(data)}. Widen the date range (MA50 needs ~50 sessions; "
        "MA200 needs ~200)."
    )
    st.stop()

if len(data) < 2:
    st.error("❌ Need at least two sessions to compute a daily change.")
    st.stop()

company_name = info.get("longName") or info.get("shortName") or ticker
sector = info.get("sector") or "N/A"
industry = info.get("industry") or "N/A"

st.title(f"📈 {company_name}")
st.caption(
    f"**Ticker:** {ticker}  |  **Sector:** {sector}  |  **Industry:** {industry}  |  "
    f"**Bars:** {len(data)}  |  **Source:** {source_label}"
)
if source_id == "yahoo":
    st.warning(
        "Yahoo Finance is **not exchange data** (delayed unofficial scrape). "
        "India names should come from NSE; global names from Stooq. "
        "This path is only used when those feeds fail."
    )
elif source_id == "demo":
    st.warning("Demo series — synthetic prices, **not** a market. Uncheck Offline demo / fix network for real bars.")
elif source_id == "nse" and live:
    st.success("NSE India official feed. Last bar is spliced with **last traded price** when the session is today.")
elif source_note:
    st.info(source_note)
st.markdown(
    '<div class="warn-box">Educational simulation — not a live broker, not advice. '
    "Paper fills use last close/LTP + slippage. The backtest fills at <b>next open</b> "
    "with discount-broker fees. Do not trade real money from these numbers.</div>",
    unsafe_allow_html=True,
)

data = add_technical_indicators(data)

price = last_valid(data["Close"])
prev_price = last_valid(data["Close"], 2)
if not (price == price) or not (prev_price == prev_price) or prev_price == 0:
    st.error("❌ Last closes are unusable. Try a different range.")
    st.stop()

day_chg = price - prev_price
day_pct = (day_chg / prev_price) * 100
if quote.get("previousClose"):
    prev_pc = float(quote["previousClose"])
    if prev_pc:
        prev_price = prev_pc
        day_chg = price - prev_price
        day_pct = float(quote["pChange"]) if quote.get("pChange") is not None else (day_chg / prev_price) * 100

def _lv(col: str) -> float:
    return last_valid(data[col]) if col in data.columns else float("nan")

rsi_val = _lv("RSI")
macd_val = _lv("MACD")
macd_sig = _lv("MACD_S")
macd_hist = _lv("MACD_H")
atr_val = _lv("ATR")
adx_val = _lv("ADX")
cci_val = _lv("CCI")
willr_val = _lv("WILLR")
obv_val = _lv("OBV")
stoch_k = _lv("STOCH_K")
stoch_d = _lv("STOCH_D")
bb_u = _lv("BB_U")
bb_l = _lv("BB_L")
bb_m = _lv("BB_M")
sym = money_symbol(ticker)

n_hi = min(252, len(data))
hi = float(data["High"].tail(n_hi).max())
lo = float(data["Low"].tail(n_hi).min())
hi_label = "52W High" if n_hi >= 252 else f"{n_hi}D High"
lo_label = "52W Low" if n_hi >= 252 else f"{n_hi}D Low"

st.markdown("---")
r1c1, r1c2, r1c3 = st.columns(3)
price_label = "💰 NSE LTP" if (source_id == "nse" and live) else "💰 Last close"
if quote.get("pChange") is not None and source_id == "nse":
    delta = f"{quote.get('change', day_chg):+.2f} ({float(quote['pChange']):+.2f}%)"
else:
    delta = f"{day_chg:+.2f} ({day_pct:+.2f}%)"
r1c1.metric(price_label, fmt_money(price, ticker), delta)
r1c2.metric(f"📈 {hi_label}", fmt_money(hi, ticker))
r1c3.metric(f"📉 {lo_label}", fmt_money(lo, ticker))
r2c1, r2c2, r2c3 = st.columns(3)
r2c1.metric("📊 RSI (14)", f"{rsi_val:.1f}" if rsi_val == rsi_val else "—")
r2c2.metric("⚡ ATR", f"{atr_val:.2f}" if atr_val == atr_val else "—")
r2c3.metric("🔊 Volume", fmt_volume(data["Volume"].iloc[-1]))
st.markdown("---")

# ─────────────────────────────────────────────
# SIGNAL + TICKET
# ─────────────────────────────────────────────
signals, score = compute_signal_score(data, rsi_ob, rsi_os)
signal_label, signal_cls, signal_emoji = classify_signal(score)

st.subheader("🤖 Study signal + paper ticket")
left, right = st.columns([1, 2])

with left:
    st.markdown(
        f"""
        <div class="signal-card {signal_cls}">
            {signal_emoji} {esc(signal_label)}<br>
            <span style="font-size:14px;opacity:.8">Composite score: {score:+d} / 100</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left_pct = 50 + min(score, 0) / 2
    width_pct = abs(score) / 2
    bar_color = "#52b788" if score > 0 else ("#f87171" if score < 0 else "#fbbf24")
    st.markdown(
        f"""
        <div style="position:relative;background:#2d3748;border-radius:6px;height:14px;width:100%">
          <div style="position:absolute;left:50%;top:0;width:2px;height:14px;background:#9ca3af;"></div>
          <div style="position:absolute;left:{left_pct}%;width:{width_pct}%;height:14px;
                      background:{bar_color};border-radius:6px;"></div>
        </div>
        <p style="color:#9ca3af;font-size:12px;margin-top:4px">−100 sell &nbsp;|&nbsp; 0 &nbsp;|&nbsp; +100 buy</p>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown("**📋 Indicator breakdown**")
    for name, (label, tone, pts) in signals.items():
        arrow = "▲" if pts > 0 else ("▼" if pts < 0 else "─")
        st.markdown(
            f"""
            <div class="indicator-row">
              <span class="metric-label">{esc(name)}</span>
              <span class="{tone}">{arrow} {esc(label)}</span>
              <span class="metric-value">{pts:+d}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

book = st.session_state.paper_book
ccy = currency_for(ticker)
prices_now = {ticker: price}
eq = equity_by_ccy(book, prices_now)
held = int(book["positions"].get(ticker, {}).get("qty", 0) or 0)
plan = suggest_trade(
    ticker, price, atr_val, score,
    equity=eq[ccy], cash=book["cash"][ccy],
    held_qty=held, risk_pct=risk_pct,
)

st.markdown("##### 🧾 Simulated order ticket")
st.markdown(
    f"""
    <div class="ticket-box">
      Suggested: <b>{esc(plan['side'])}</b> &nbsp;·&nbsp;
      Qty <b>{plan['qty']}</b> &nbsp;·&nbsp;
      Est. fill <b>{fmt_money(plan['fill'], ticker)}</b>
      (last close + slippage) &nbsp;·&nbsp;
      Stop <b>{fmt_money(plan['stop'], ticker)}</b> (1.5×ATR) &nbsp;·&nbsp;
      Target <b>{fmt_money(plan['target'], ticker)}</b> (2R) &nbsp;·&nbsp;
      Fees <b>{fmt_money(plan['fee'], ticker)}</b><br>
      <span style="color:#9ca3af;font-size:12px">
        {esc(plan['model_name'])} · risk {risk_pct*100:.2f}% of {ccy} equity
        ({fmt_money(eq[ccy], ticker)}) · cash {fmt_money(book['cash'][ccy], ticker)}
        · held {held} sh
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

qty_in = st.number_input("Quantity", min_value=0, value=int(plan["qty"]), step=1)
t1, t2, t3, t4 = st.columns(4)
with t1:
    buy_clicked = st.button("BUY (paper)", type="primary", width="stretch")
with t2:
    sell_clicked = st.button("SELL (paper)", width="stretch")
with t3:
    st.caption(f"INR cash {book['cash']['INR']:,.0f}")
with t4:
    st.caption(f"USD cash {book['cash']['USD']:,.2f}")

if buy_clicked:
    new_book, msg = execute(
        book, ticker, "BUY", int(qty_in), price,
        stop=plan["stop"], target=plan["target"],
    )
    if msg == "ok":
        st.session_state.paper_book = new_book
        st.success(f"Paper BUY {qty_in} {ticker} @ ~{fmt_money(plan['fill'], ticker)}")
        st.rerun()
    else:
        st.error(msg)

if sell_clicked:
    new_book, msg = execute(book, ticker, "SELL", int(qty_in), price)
    if msg == "ok":
        st.session_state.paper_book = new_book
        st.success(f"Paper SELL {ticker}")
        st.rerun()
    else:
        st.error(msg)

if book["positions"]:
    pos_rows = []
    for t, p in book["positions"].items():
        px = price if t == ticker else float(p.get("avg", 0))
        qty_p = float(p["qty"])
        mtm = qty_p * px
        pos_rows.append(
            {
                "Ticker": t,
                "Qty": int(qty_p),
                "Avg": round(float(p["avg"]), 2),
                "Last": round(px, 2),
                "MTM": round(mtm, 2),
                "Stop": p.get("stop"),
                "Target": p.get("target"),
            }
        )
    st.dataframe(pd.DataFrame(pos_rows), width="stretch", hide_index=True)

if book["trades"]:
    with st.expander("Paper trade log"):
        st.dataframe(pd.DataFrame(book["trades"][::-1]), width="stretch", hide_index=True)

st.markdown("---")

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
st.subheader("📊 Price chart")
st.plotly_chart(build_candlestick_chart(data, show_bb), width="stretch")

if show_macd or show_rsi:
    if show_macd and show_rsi:
        col_macd, col_rsi = st.columns(2)
        with col_macd:
            st.markdown("**MACD**")
            st.plotly_chart(build_macd_chart(data), width="stretch")
        with col_rsi:
            st.markdown("**RSI (14)**")
            st.plotly_chart(build_rsi_chart(data, rsi_ob, rsi_os), width="stretch")
    elif show_macd:
        st.markdown("**MACD**")
        st.plotly_chart(build_macd_chart(data), width="stretch")
    else:
        st.markdown("**RSI (14)**")
        st.plotly_chart(build_rsi_chart(data, rsi_ob, rsi_os), width="stretch")

st.markdown("---")

# ─────────────────────────────────────────────
# ML
# ─────────────────────────────────────────────
st.subheader("🧠 Next-session close forecast")
st.caption(
    "Target is **tomorrow’s close**. Features are only values known at **today’s close**. "
    "Persistence = “tomorrow ≈ today”. If ML cannot beat it, there is no edge."
)

data_ml = add_ml_features(data)
results = train_models(data_ml, FEATURES, k)

pred_avg = float("nan")
if results.get("error"):
    st.warning(results["error"])
    perf_df = pd.DataFrame()
    X_train_len = X_test_len = 0
else:
    perf_df = results["perf_df"]
    X_train_len = len(results["X_train"])
    X_test_len = len(results["X_test"])
    st.markdown("**Model comparison (test window)**")
    try:
        styler = perf_df.style.highlight_min(subset=["RMSE", "MAPE %"], color="#1a472a")
        if "Dir. Acc %" in perf_df.columns:
            styler = styler.highlight_max(subset=["Dir. Acc %"], color="#1a472a")
        st.dataframe(styler, width="stretch")
    except Exception:
        st.dataframe(perf_df, width="stretch")

    if not results.get("beats_persistence", True):
        st.info("No ML model beat persistence RMSE on this split. Treat the price forecast as a coin flip.")

    st.plotly_chart(
        build_prediction_chart(results["y_test"], results["predictions"]),
        width="stretch",
    )

    preds, pred_avg, cur_sym = predict_next_day(results, data_ml, FEATURES, price, ticker)
    st.markdown("**Forecast for the next session close**")
    names = ["Persistence", "KNN", "Ridge", "Random Forest", "Gradient Boosting"]
    cols = st.columns(len(names) + 1)
    for i, name in enumerate(names):
        if name not in preds:
            continue
        p = preds[name]
        cols[i].metric(name, f"{cur_sym}{p:.2f}", f"{((p - price) / price) * 100:+.2f}%")
    if pred_avg == pred_avg:
        cols[-1].metric(
            "🎯 ML ensemble",
            f"{cur_sym}{pred_avg:.2f}",
            f"{((pred_avg - price) / price) * 100:+.2f}%",
        )

st.markdown("---")

# ─────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────
st.subheader("📉 Signal backtest (next-open fills)")
st.caption(
    "Each close produces a BUY / SELL / HOLD. Orders fill at the **following open** "
    "with slippage and fees. Stops / targets use that day’s high–low. Compared with buy & hold."
)

initial = DEFAULT_CASH_INR if ccy == "INR" else DEFAULT_CASH_USD
with st.spinner("Running walk-forward-style EOD backtest…"):
    bt = run_signal_backtest(
        data, rsi_ob, rsi_os, ticker,
        initial_capital=initial,
        long_short=long_short,
        risk_pct=risk_pct,
    )

if bt.get("error"):
    st.warning(bt["error"])
else:
    s = bt["stats"]
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("Strategy", f"{s['return_pct']:+.1f}%")
    b2.metric("Buy & hold", f"{s['buy_hold_pct']:+.1f}%")
    b3.metric("CAGR", f"{s['cagr_pct']:+.1f}%" if s["cagr_pct"] == s["cagr_pct"] else "—")
    b4.metric("Max DD", f"{s['max_dd_pct']:.1f}%")
    b5.metric("Sharpe", f"{s['sharpe']:.2f}" if s["sharpe"] == s["sharpe"] else "—")
    wr = s["win_rate_pct"]
    b6.metric("Win rate", f"{wr:.0f}%" if wr == wr else "—", f"{s['round_trips']} round trips")
    st.caption(f"Costs: {s['cost_name']} · {s['trades']} fills · {s['bars']} sessions")
    st.plotly_chart(build_equity_chart(bt["equity"], bt["buy_hold"]), width="stretch")
    if bt["trades"] is not None and not bt["trades"].empty:
        with st.expander("Backtest fills"):
            st.dataframe(bt["trades"].tail(50), width="stretch", hide_index=True)

st.markdown("---")

# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────
st.subheader("📝 Auto analysis report")
info_vals = dict(
    price=price, day_chg=day_chg, day_pct=day_pct,
    rsi_val=safe_float(rsi_val, 0.0), macd_val=safe_float(macd_val, 0.0),
    macd_sig=safe_float(macd_sig, 0.0), macd_hist=safe_float(macd_hist, 0.0),
    stoch_k=safe_float(stoch_k, 0.0), stoch_d=safe_float(stoch_d, 0.0),
    adx_val=safe_float(adx_val, 0.0), cci_val=safe_float(cci_val, 0.0),
    willr_val=safe_float(willr_val, 0.0), atr_val=safe_float(atr_val, 0.0),
    obv_val=safe_float(obv_val, 0.0),
    bb_u=safe_float(bb_u, price), bb_l=safe_float(bb_l, price), bb_m=safe_float(bb_m, price),
    rsi_ob=rsi_ob, rsi_os=rsi_os,
)
signal_info = dict(score=score, signal_label=signal_label)
ml_info = dict(
    perf_df=perf_df, pred_avg=pred_avg,
    X_train_len=X_train_len, X_test_len=X_test_len,
)
st.markdown(
    generate_report(data, ticker, company_name, info_vals, signal_info, ml_info),
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────
# HEATMAP
# ─────────────────────────────────────────────
st.subheader("🌍 Market heatmap (~5-session return, NSE LTP / Stooq)")
with st.spinner("Loading heatmap…"):
    heat = load_heatmap_returns(tuple(ALL_TICKERS.keys()))
st.plotly_chart(build_heatmap(heat, ALL_TICKERS), width="stretch")

# ─────────────────────────────────────────────
# RAW DATA
# ─────────────────────────────────────────────
with st.expander("📄 View raw data"):
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "MA20", "MA50", "RSI", "MACD", "ATR"] if c in data.columns]
    st.dataframe(data[cols].tail(30).round(2), width="stretch")

st.markdown(
    """
    <hr>
    <p style="text-align:center;color:#6b7280;font-size:13px">
      Built with Streamlit · NSE India · Stooq · scikit-learn · Plotly.
      Yahoo is a last-resort fallback only. Paper broker is not execution.
    </p>
    """,
    unsafe_allow_html=True,
)
