"""
Auto-generated HTML analysis report.
"""

import numpy as np
import pandas as pd
from datetime import datetime


def _last(data, col):
    return float(data[col].iloc[-1])


def generate_report(data, ticker, company_name, info_vals, signal_info, ml_info):
    """
    Build the full HTML analysis report.

    Parameters
    ----------
    data : DataFrame with all indicators computed.
    ticker : str
    company_name : str
    info_vals : dict with price, day_chg, day_pct, rsi_val, macd_val,
                macd_sig, macd_hist, stoch_k, stoch_d, adx_val, cci_val,
                willr_val, atr_val, obv_val, bb_u, bb_l, bb_m, rsi_ob, rsi_os
    signal_info : dict with score, signal_label
    ml_info : dict with perf_df, pred_avg, X_train_len, X_test_len
    """
    v = info_vals
    price = v["price"]
    day_chg = v["day_chg"]
    day_pct = v["day_pct"]
    rsi_val = v["rsi_val"]
    macd_val = v["macd_val"]
    macd_sig = v["macd_sig"]
    macd_hist = v["macd_hist"]
    stoch_k = v["stoch_k"]
    stoch_d = v["stoch_d"]
    adx_val = v["adx_val"]
    cci_val = v["cci_val"]
    willr_val = v["willr_val"]
    atr_val = v["atr_val"]
    obv_val = v["obv_val"]
    bb_u = v["bb_u"]
    bb_l = v["bb_l"]
    bb_m = v["bb_m"]
    rsi_ob = v["rsi_ob"]
    rsi_os = v["rsi_os"]

    score = signal_info["score"]
    signal_label = signal_info["signal_label"]
    perf_df = ml_info["perf_df"]
    pred_avg = ml_info["pred_avg"]

    cur_sym = "₹" if ".NS" in ticker else "$"
    close = data["Close"].squeeze()
    volume = data["Volume"].squeeze()

    # Support / Resistance
    recent = data.tail(60)
    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())
    support_20 = float(data["Low"].tail(20).min())
    resistance_20 = float(data["High"].tail(20).max())

    # Volatility regime
    hist_vol = float(close.pct_change().tail(20).std() * np.sqrt(252) * 100)
    vol_regime = ("High Volatility" if hist_vol > 40 else
                  "Moderate Volatility" if hist_vol > 20 else "Low Volatility")

    # Volume analysis
    avg_vol_20 = float(volume.tail(20).mean())
    last_vol = float(volume.iloc[-1])
    vol_vs_avg = ((last_vol - avg_vol_20) / avg_vol_20) * 100
    vol_note = "above" if vol_vs_avg > 0 else "below"

    # Trend detection
    trend_20 = "Uptrend" if float(data["MA20"].iloc[-1]) > float(data["MA20"].iloc[-20]) else "Downtrend"
    trend_50 = "Uptrend" if float(data["MA50"].iloc[-1]) > float(data["MA50"].iloc[-50]) else "Downtrend"
    trend_200 = "Uptrend" if float(data["MA200"].iloc[-1]) > float(data["MA200"].iloc[-60]) else "Downtrend"

    # Price position
    high_52 = float(data["High"].tail(252).max())
    low_52 = float(data["Low"].tail(252).min())
    pct_from_52h = ((price - high_52) / high_52) * 100
    pct_from_52l = ((price - low_52) / low_52) * 100

    # Best model
    best_model = perf_df["MAPE %"].idxmin()
    best_mape = perf_df["MAPE %"].min()

    X_train_len = ml_info["X_train_len"]
    X_test_len = ml_info["X_test_len"]

    # Trend alignment text
    if trend_20 == trend_50 == trend_200 == "Uptrend":
        trend_text = "All three timeframes align bullishly — a strong signal."
    elif trend_20 == trend_50 == trend_200 == "Downtrend":
        trend_text = "All three timeframes align bearishly — caution warranted."
    else:
        trend_text = "Trend is mixed across timeframes — wait for clarity."

    # Assessment text
    if score >= 35:
        assess = ("The confluence of bullish indicators across momentum, trend, and volume "
                  "suggests favourable risk/reward for long positions. Consider scaling in near support.")
    elif score <= -35:
        assess = ("Multiple indicators confirm selling pressure. Risk management is critical "
                  "— consider reducing exposure or setting tight stops.")
    elif score > 0:
        assess = ("Slight bullish bias but not yet confirmed. Wait for a breakout above "
                  "resistance or further indicator convergence before acting.")
    elif score < 0:
        assess = ("Slight bearish bias. Monitor closely. A break below support may "
                  "confirm downside momentum.")
    else:
        assess = "The market is in a neutral, indecisive phase. Range-trading strategies may be appropriate."

    report_html = f"""
<div class="report-box">
  <h3 style="color:#60a5fa;margin-top:0">📊 {company_name} — Automated Analysis Report</h3>
  <p style="color:#9ca3af;font-size:13px">Generated: {datetime.now().strftime('%d %b %Y %H:%M')} &nbsp;|&nbsp; Ticker: {ticker}</p>
  <hr style="border-color:#374151">

  <h4 style="color:#fbbf24">1. Price Summary</h4>
  <p>
    The stock is currently trading at <b style="color:#f3f4f6">{cur_sym}{price:,.2f}</b>,
    {"gaining" if day_chg >= 0 else "losing"}
    <b style="color:{'#52b788' if day_chg >= 0 else '#f87171'}">{abs(day_chg):.2f} ({abs(day_pct):.2f}%)</b>
    on the day.
    It is <b>{abs(pct_from_52h):.1f}%</b> below its 52-week high and
    <b>{pct_from_52l:.1f}%</b> above its 52-week low.
    Near-term support is at <b>{cur_sym}{support_20:,.2f}</b> and
    resistance at <b>{cur_sym}{resistance_20:,.2f}</b>.
    The 60-day range shows support at <b>{cur_sym}{support:,.2f}</b> and
    resistance at <b>{cur_sym}{resistance:,.2f}</b>.
  </p>

  <h4 style="color:#fbbf24">2. Trend Analysis</h4>
  <p>
    Short-term (MA20): <b class="{'bullish' if trend_20=='Uptrend' else 'bearish'}">{trend_20}</b> &nbsp;|&nbsp;
    Medium-term (MA50): <b class="{'bullish' if trend_50=='Uptrend' else 'bearish'}">{trend_50}</b> &nbsp;|&nbsp;
    Long-term (MA200): <b class="{'bullish' if trend_200=='Uptrend' else 'bearish'}">{trend_200}</b>.
    {trend_text}
    The ADX reads <b>{adx_val:.1f}</b>, indicating a
    <b>{"strong" if adx_val > 25 else "moderate" if adx_val > 20 else "weak"}</b> trend.
  </p>

  <h4 style="color:#fbbf24">3. Momentum Indicators</h4>
  <p>
    <b>RSI</b> at <b>{rsi_val:.1f}</b> is
    <b class="{'bearish' if rsi_val > rsi_ob else 'bullish' if rsi_val < rsi_os else 'neutral'}">
      {"overbought" if rsi_val > rsi_ob else "oversold" if rsi_val < rsi_os else "neutral"}</b>.
    <b>MACD</b> is <b class="{'bullish' if macd_val > macd_sig else 'bearish'}">
      {"above" if macd_val > macd_sig else "below"}</b> its signal line
    (MACD: {macd_val:.3f}, Signal: {macd_sig:.3f}), histogram: {macd_hist:.3f}.
    <b>Stochastic</b> K={stoch_k:.1f} D={stoch_d:.1f} —
    <b class="{'bullish' if stoch_k < 20 else 'bearish' if stoch_k > 80 else 'neutral'}">
      {"oversold" if stoch_k < 20 else "overbought" if stoch_k > 80 else "neutral"}</b>.
    <b>CCI</b> at {cci_val:.1f} —
    <b class="{'bullish' if cci_val < -100 else 'bearish' if cci_val > 100 else 'neutral'}">
      {"oversold" if cci_val < -100 else "overbought" if cci_val > 100 else "neutral"}</b>.
    <b>Williams %R</b> at {willr_val:.1f} —
    <b class="{'bullish' if willr_val < -80 else 'bearish' if willr_val > -20 else 'neutral'}">
      {"oversold" if willr_val < -80 else "overbought" if willr_val > -20 else "neutral"}</b>.
  </p>

  <h4 style="color:#fbbf24">4. Volatility &amp; Volume</h4>
  <p>
    Annualised historical volatility (20d) is <b>{hist_vol:.1f}%</b> —
    <b class="neutral">{vol_regime}</b>.
    ATR is <b>{atr_val:.2f}</b>, implying expected daily range of
    roughly <b>{(atr_val / price) * 100:.2f}%</b>.
    Bollinger Bands show price is
    <b class="{'bearish' if price > bb_u else 'bullish' if price < bb_l else 'neutral'}">
      {"above upper band (extended)" if price > bb_u else "below lower band (compressed)" if price < bb_l else "within bands"}</b>.
    Last session volume was <b>{last_vol:,.0f}</b>, which is
    <b class="{'bullish' if vol_vs_avg > 10 else 'bearish' if vol_vs_avg < -10 else 'neutral'}">
      {abs(vol_vs_avg):.1f}% {vol_note} the 20-day average</b>.
    OBV is <b class="{'bullish' if obv_val > 0 else 'bearish'}">
      {"rising (accumulation)" if obv_val > float(data["OBV"].iloc[-5]) else "falling (distribution)"}</b>.
  </p>

  <h4 style="color:#fbbf24">5. ML Prediction Summary</h4>
  <p>
    Four models were trained on {X_train_len} samples and tested on {X_test_len}.
    The best-performing model is <b>{best_model}</b> with a MAPE of <b>{best_mape:.2f}%</b>.
    The ensemble average next-day forecast is
    <b style="color:#60a5fa">{cur_sym}{pred_avg:.2f}</b>
    (<b class="{'bullish' if pred_avg > price else 'bearish'}">
      {((pred_avg - price) / price) * 100:+.2f}%</b> from current price).
  </p>

  <h4 style="color:#fbbf24">6. Overall Assessment</h4>
  <p>
    The composite AI signal score is <b style="color:{'#52b788' if score > 0 else '#f87171' if score < 0 else '#fbbf24'}">{score:+d}/100</b>,
    corresponding to a <b>{signal_label}</b> recommendation.
    {assess}
  </p>

  <hr style="border-color:#374151">
  <p style="color:#6b7280;font-size:12px">
    ⚠️ <b>Disclaimer:</b> This report is generated algorithmically for educational purposes only.
    It does not constitute financial advice. Always do your own research before investing.
  </p>
</div>
"""
    return report_html
