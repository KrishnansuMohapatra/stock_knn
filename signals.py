"""
Trading signal engine — composite scoring from multiple indicators.

Scores run from -100 (strong sell) to +100 (strong buy).
MACD crossover is an *event* (previous bar vs this bar), not a state.
ADX is signed by +DI vs −DI (direction), not strength alone.
"""

from __future__ import annotations

import math

import pandas as pd


def _num(data: pd.DataFrame, col: str, i: int) -> float:
    if col not in data.columns or i < 0 or i >= len(data):
        return float("nan")
    val = data[col].iloc[i]
    try:
        f = float(val)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(f) or math.isinf(f):
        return float("nan")
    return f


def _finite(*vals: float) -> bool:
    return all(not (math.isnan(v) or math.isinf(v)) for v in vals)


def compute_signal_score(data: pd.DataFrame, rsi_ob: int, rsi_os: int, i: int | None = None):
    """
    Score a single bar. ``i=None`` means the last row.

    Returns
    -------
    signals : dict[str, tuple[str, str, int]]
        indicator name → (label, tone, points)
    total : int
        Clamped composite score.
    """
    if data is None or data.empty:
        return {}, 0

    if i is None:
        i = len(data) - 1
    if i < 0:
        i = len(data) + i
    if i < 1 or i >= len(data):
        return {}, 0

    rsi_val = _num(data, "RSI", i)
    macd_val = _num(data, "MACD", i)
    macd_sig = _num(data, "MACD_S", i)
    macd_prev = _num(data, "MACD", i - 1)
    macd_sig_prev = _num(data, "MACD_S", i - 1)
    price = _num(data, "Close", i)
    ma20 = _num(data, "MA20", i)
    ma50 = _num(data, "MA50", i)
    ma200 = _num(data, "MA200", i)
    bb_u = _num(data, "BB_U", i)
    bb_l = _num(data, "BB_L", i)
    bb_m = _num(data, "BB_M", i)
    stoch_k = _num(data, "STOCH_K", i)
    stoch_d = _num(data, "STOCH_D", i)
    adx_val = _num(data, "ADX", i)
    plus_di = _num(data, "PLUS_DI", i)
    minus_di = _num(data, "MINUS_DI", i)
    cci_val = _num(data, "CCI", i)
    willr_val = _num(data, "WILLR", i)

    signals: dict[str, tuple[str, str, int]] = {}

    # ── RSI ──────────────────────────────────
    if not _finite(rsi_val):
        signals["RSI"] = ("Unavailable", "neutral", 0)
    elif rsi_val < rsi_os:
        signals["RSI"] = ("Oversold – Bullish", "bullish", +15)
    elif rsi_val > rsi_ob:
        signals["RSI"] = ("Overbought – Bearish", "bearish", -15)
    elif 40 <= rsi_val <= 60:
        signals["RSI"] = ("Neutral Zone", "neutral", 0)
    elif rsi_val >= 60:
        signals["RSI"] = ("Mildly Bullish", "bullish", +7)
    else:
        signals["RSI"] = ("Mildly Bearish", "bearish", -7)

    # ── MACD (crossover = event) ─────────────
    if not _finite(macd_val, macd_sig, macd_prev, macd_sig_prev):
        signals["MACD"] = ("Unavailable", "neutral", 0)
    else:
        bull_cross = macd_prev <= macd_sig_prev and macd_val > macd_sig
        bear_cross = macd_prev >= macd_sig_prev and macd_val < macd_sig
        if bull_cross:
            signals["MACD"] = ("Bullish Crossover", "bullish", +20)
        elif bear_cross:
            signals["MACD"] = ("Bearish Crossover", "bearish", -20)
        elif macd_val > macd_sig:
            signals["MACD"] = ("Above signal", "bullish", +8)
        else:
            signals["MACD"] = ("Below signal", "bearish", -8)

    # ── Moving Averages ───────────────────────
    if not _finite(price, ma20, ma50):
        signals["Moving Averages"] = ("Unavailable", "neutral", 0)
    else:
        checks = [price > ma20, price > ma50, ma20 > ma50]
        if _finite(ma200):
            checks.extend([price > ma200, ma50 > ma200])
        ma_bull = sum(1 for c in checks if c)
        n = len(checks)
        ma_score = int(round(((ma_bull / n) - 0.5) * 40))
        if ma_bull >= n - 1:
            ma_label, ma_tone = "Bullish Alignment", "bullish"
        elif ma_bull <= 1:
            ma_label, ma_tone = "Bearish Alignment", "bearish"
        else:
            ma_label, ma_tone = "Mixed", "neutral"
        signals["Moving Averages"] = (
            f"{ma_label} ({ma_bull}/{n} bullish)",
            ma_tone,
            ma_score,
        )

    # ── Bollinger Bands ───────────────────────
    if not _finite(price, bb_u, bb_l, bb_m):
        signals["Bollinger Bands"] = ("Unavailable", "neutral", 0)
    elif price < bb_l:
        signals["Bollinger Bands"] = ("Below lower band – oversold", "bullish", +12)
    elif price > bb_u:
        signals["Bollinger Bands"] = ("Above upper band – overbought", "bearish", -12)
    elif price > bb_m:
        signals["Bollinger Bands"] = ("Above mid-band", "bullish", +5)
    else:
        signals["Bollinger Bands"] = ("Below mid-band", "bearish", -5)

    # ── Stochastic ───────────────────────────
    if not _finite(stoch_k, stoch_d):
        signals["Stochastic"] = ("Unavailable", "neutral", 0)
    elif stoch_k < 20 and stoch_k > stoch_d:
        signals["Stochastic"] = ("Oversold + bullish cross", "bullish", +12)
    elif stoch_k > 80 and stoch_k < stoch_d:
        signals["Stochastic"] = ("Overbought + bearish cross", "bearish", -12)
    elif stoch_k < 20:
        signals["Stochastic"] = ("Oversold", "bullish", +8)
    elif stoch_k > 80:
        signals["Stochastic"] = ("Overbought", "bearish", -8)
    elif stoch_k > stoch_d:
        signals["Stochastic"] = ("%K above %D", "bullish", +5)
    else:
        signals["Stochastic"] = ("%K below %D", "bearish", -5)

    # ── ADX signed by direction ───────────────
    if not _finite(adx_val, plus_di, minus_di):
        signals["ADX"] = ("Unavailable", "neutral", 0)
    else:
        up = plus_di >= minus_di
        if adx_val > 25:
            if up:
                signals["ADX"] = (f"Strong uptrend ({adx_val:.1f})", "bullish", +8)
            else:
                signals["ADX"] = (f"Strong downtrend ({adx_val:.1f})", "bearish", -8)
        elif adx_val > 20:
            if up:
                signals["ADX"] = (f"Developing uptrend ({adx_val:.1f})", "bullish", +3)
            else:
                signals["ADX"] = (f"Developing downtrend ({adx_val:.1f})", "bearish", -3)
        else:
            signals["ADX"] = (f"Weak / ranging ({adx_val:.1f})", "neutral", 0)

    # ── CCI ───────────────────────────────────
    if not _finite(cci_val):
        signals["CCI"] = ("Unavailable", "neutral", 0)
    elif cci_val < -100:
        signals["CCI"] = ("Oversold – Bullish", "bullish", +8)
    elif cci_val > 100:
        signals["CCI"] = ("Overbought – Bearish", "bearish", -8)
    else:
        signals["CCI"] = ("Neutral", "neutral", 0)

    # ── Williams %R ───────────────────────────
    if not _finite(willr_val):
        signals["Williams %R"] = ("Unavailable", "neutral", 0)
    elif willr_val < -80:
        signals["Williams %R"] = ("Oversold – Bullish", "bullish", +8)
    elif willr_val > -20:
        signals["Williams %R"] = ("Overbought – Bearish", "bearish", -8)
    else:
        signals["Williams %R"] = ("Neutral", "neutral", 0)

    # ── OBV Trend ─────────────────────────────
    lookback = 5 if i >= 5 else 1
    obv_now = _num(data, "OBV", i)
    obv_prev = _num(data, "OBV", i - lookback)
    if not _finite(obv_now, obv_prev):
        signals["OBV"] = ("Unavailable", "neutral", 0)
    else:
        slope = obv_now - obv_prev
        if slope > 0:
            signals["OBV"] = ("Rising – buying pressure", "bullish", +10)
        elif slope < 0:
            signals["OBV"] = ("Falling – selling pressure", "bearish", -10)
        else:
            signals["OBV"] = ("Flat", "neutral", 0)

    total = sum(v[2] for v in signals.values())
    total = max(-100, min(100, int(total)))
    return signals, total


def classify_signal(score: int):
    """Map a composite score to a label, CSS class and emoji."""
    if score >= 35:
        return "STRONG BUY", "buy-card", "🟢🟢"
    if score >= 15:
        return "BUY", "buy-card", "🟢"
    if score <= -35:
        return "STRONG SELL", "sell-card", "🔴🔴"
    if score <= -15:
        return "SELL", "sell-card", "🔴"
    return "HOLD / NEUTRAL", "hold-card", "🟡"


def first_signal_index(data: pd.DataFrame) -> int | None:
    """Earliest bar where core indicators are present (MA200 not required)."""
    cols = ["RSI", "MACD", "MACD_S", "ATR", "MA20", "MA50", "STOCH_K", "STOCH_D", "ADX"]
    missing = [c for c in cols if c not in data.columns]
    if missing or data.empty:
        return None
    mask = data[cols].notna().all(axis=1)
    idx = mask.to_numpy().nonzero()[0]
    if len(idx) == 0:
        return None
    start = int(idx[0])
    return start if start >= 1 else (1 if len(data) > 1 else None)
