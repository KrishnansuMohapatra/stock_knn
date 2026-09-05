"""
Shared helpers: money formatting, ticker currency, sanitisation.
"""

from __future__ import annotations

import math
from html import escape as html_escape

import pandas as pd

INR_SUFFIXES = (".NS", ".BO", ".BSE")


def currency_for(ticker: str) -> str:
    t = (ticker or "").upper()
    if t.endswith(INR_SUFFIXES):
        return "INR"
    return "USD"


def money_symbol(ticker: str) -> str:
    return "₹" if currency_for(ticker) == "INR" else "$"


def fmt_money(value, ticker: str, decimals: int = 2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(value) or math.isinf(value):
        return "—"
    return f"{money_symbol(ticker)}{value:,.{decimals}f}"


def fmt_volume(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(n) or math.isinf(n):
        return "—"
    abs_n = abs(n)
    if abs_n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if abs_n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs_n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return f"{int(n):,}"


def safe_float(val, default=float("nan")) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def last_valid(series: pd.Series, n: int = 1) -> float:
    if series is None or len(series) == 0:
        return float("nan")
    s = series.dropna()
    if len(s) < n:
        return float("nan")
    return float(s.iloc[-n])


def esc(text) -> str:
    return html_escape("" if text is None else str(text), quote=True)


def is_finite(val) -> bool:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(f) or math.isinf(f))
