"""
Trading signal engine – composite scoring from multiple indicators.
"""

import pandas as pd


def _last(data: pd.DataFrame, col: str) -> float:
    return float(data[col].iloc[-1])


def _prev(data: pd.DataFrame, col: str) -> float:
    return float(data[col].iloc[-2])


def compute_signal_score(data: pd.DataFrame, rsi_ob: int, rsi_os: int):
    """
    Score ranges from -100 (strong sell) to +100 (strong buy).
    Each indicator votes and votes are weighted.

    Returns
    -------
    signals : dict[str, tuple[str, str, int]]
        Mapping of indicator name → (label, tone, points).
    total : int
        Clamped composite score.
    """
    rsi_val = _last(data, "RSI")
    macd_val = _last(data, "MACD")
    macd_sig = _last(data, "MACD_S")
    macd_hist = _last(data, "MACD_H")
    price = _last(data, "Close")
    ma20 = _last(data, "MA20")
    ma50 = _last(data, "MA50")
    ma200 = _last(data, "MA200")
    bb_u = _last(data, "BB_U")
    bb_l = _last(data, "BB_L")
    bb_m = _last(data, "BB_M")
    stoch_k = _last(data, "STOCH_K")
    stoch_d = _last(data, "STOCH_D")
    adx_val = _last(data, "ADX")
    cci_val = _last(data, "CCI")
    willr_val = _last(data, "WILLR")

    signals = {}

    # ── RSI ──────────────────────────────────
    if rsi_val < rsi_os:
        signals["RSI"] = ("Oversold – Bullish", "bullish", +15)
    elif rsi_val > rsi_ob:
        signals["RSI"] = ("Overbought – Bearish", "bearish", -15)
    elif 40 <= rsi_val <= 60:
        signals["RSI"] = ("Neutral Zone", "neutral", 0)
    elif rsi_val >= 60:
        signals["RSI"] = ("Mildly Bullish", "bullish", +7)
    else:
        signals["RSI"] = ("Mildly Bearish", "bearish", -7)

    # ── MACD ─────────────────────────────────
    if macd_val > macd_sig and macd_hist > 0:
        signals["MACD"] = ("Bullish Crossover", "bullish", +20)
    elif macd_val < macd_sig and macd_hist < 0:
        signals["MACD"] = ("Bearish Crossover", "bearish", -20)
    elif macd_hist > _prev(data, "MACD_H"):
        signals["MACD"] = ("Histogram Rising", "bullish", +8)
    else:
        signals["MACD"] = ("Histogram Falling", "bearish", -8)

    # ── Moving Averages ───────────────────────
    ma_bull = sum([price > ma20, price > ma50, price > ma200,
                   ma20 > ma50, ma50 > ma200])
    ma_score = (ma_bull - 2.5) * 8  # -20 to +20
    ma_label = ("Bullish Alignment" if ma_bull >= 4 else
                "Bearish Alignment" if ma_bull <= 1 else "Mixed")
    ma_tone = ("bullish" if ma_bull >= 4 else
               "bearish" if ma_bull <= 1 else "neutral")
    signals["Moving Averages"] = (f"{ma_label} ({ma_bull}/5 bullish)", ma_tone, int(ma_score))

    # ── Bollinger Bands ───────────────────────
    if price < bb_l:
        signals["Bollinger Bands"] = ("Price below Lower Band – Oversold", "bullish", +12)
    elif price > bb_u:
        signals["Bollinger Bands"] = ("Price above Upper Band – Overbought", "bearish", -12)
    elif price > bb_m:
        signals["Bollinger Bands"] = ("Above Mid-Band – Slight Bullish", "bullish", +5)
    else:
        signals["Bollinger Bands"] = ("Below Mid-Band – Slight Bearish", "bearish", -5)

    # ── Stochastic ───────────────────────────
    if stoch_k < 20 and stoch_k > stoch_d:
        signals["Stochastic"] = ("Oversold + Bullish Cross", "bullish", +12)
    elif stoch_k > 80 and stoch_k < stoch_d:
        signals["Stochastic"] = ("Overbought + Bearish Cross", "bearish", -12)
    elif stoch_k > stoch_d:
        signals["Stochastic"] = ("Bullish", "bullish", +5)
    else:
        signals["Stochastic"] = ("Bearish", "bearish", -5)

    # ── ADX (Trend Strength) ──────────────────
    if adx_val > 25:
        trend_str = "Strong Trend"
        adx_pts = +5
    elif adx_val > 20:
        trend_str = "Developing Trend"
        adx_pts = +2
    else:
        trend_str = "Weak / Ranging"
        adx_pts = -2
    signals["ADX"] = (f"{trend_str} ({adx_val:.1f})", "neutral", adx_pts)

    # ── CCI ───────────────────────────────────
    if cci_val < -100:
        signals["CCI"] = ("Oversold – Bullish", "bullish", +8)
    elif cci_val > 100:
        signals["CCI"] = ("Overbought – Bearish", "bearish", -8)
    else:
        signals["CCI"] = ("Neutral", "neutral", 0)

    # ── Williams %R ───────────────────────────
    if willr_val < -80:
        signals["Williams %R"] = ("Oversold – Bullish", "bullish", +8)
    elif willr_val > -20:
        signals["Williams %R"] = ("Overbought – Bearish", "bearish", -8)
    else:
        signals["Williams %R"] = ("Neutral", "neutral", 0)

    # ── OBV Trend ─────────────────────────────
    obv_series = data["OBV"]
    obv_slope = float(obv_series.iloc[-1]) - float(obv_series.iloc[-5])
    if obv_slope > 0:
        signals["OBV"] = ("Rising – Buying Pressure", "bullish", +10)
    else:
        signals["OBV"] = ("Falling – Selling Pressure", "bearish", -10)

    total = sum(v[2] for v in signals.values())
    total = max(-100, min(100, total))
    return signals, total


def classify_signal(score: int):
    """
    Map a composite score to a human-readable label, CSS class and emoji.

    Returns
    -------
    label, css_class, emoji : tuple[str, str, str]
    """
    if score >= 35:
        return "STRONG BUY", "buy-card", "🟢🟢"
    elif score >= 15:
        return "BUY", "buy-card", "🟢"
    elif score <= -35:
        return "STRONG SELL", "sell-card", "🔴🔴"
    elif score <= -15:
        return "SELL", "sell-card", "🔴"
    else:
        return "HOLD / NEUTRAL", "hold-card", "🟡"
