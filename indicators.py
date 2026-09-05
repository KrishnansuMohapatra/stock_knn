"""
Technical indicator computation using numpy and pandas only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = 100 * (close - lowest_low) / denom
    stoch_d = stoch_k.rolling(d_window).mean()
    return stoch_k, stoch_d


def _roc(series: pd.Series, window: int = 12) -> pd.Series:
    prev = series.shift(window).replace(0, np.nan)
    return ((series - series.shift(window)) / prev) * 100


def _williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    return -100 * (highest_high - close) / denom


def _macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    middle = _sma(series, window)
    std = series.rolling(window).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower) / middle.replace(0, np.nan)
    return upper, middle, lower, width


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _vwap20(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    num = (typical_price * volume).rolling(window).sum()
    den = volume.rolling(window).sum().replace(0, np.nan)
    return num / den


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm_raw = high - prev_high
    minus_dm_raw = prev_low - low

    plus_dm = plus_dm_raw.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), 0.0)
    minus_dm = minus_dm_raw.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), 0.0)

    atr = _atr(high, low, close, window)

    smoothed_plus_dm = plus_dm.ewm(alpha=1 / window, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100 * smoothed_plus_dm / atr.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_dm / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return adx, plus_di, minus_di


def _cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    sma_tp = _sma(typical_price, window)
    mean_deviation = typical_price.rolling(window).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (typical_price - sma_tp) / (0.015 * mean_deviation.replace(0, np.nan))


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators. Early bars stay NaN (charts can plot them);
    callers must not assume the last row of MA200 exists on short histories.
    """
    out = data.copy()
    close = out["Close"].squeeze()
    high = out["High"].squeeze()
    low = out["Low"].squeeze()
    volume = out["Volume"].squeeze()

    out["MA20"] = _sma(close, 20)
    out["MA50"] = _sma(close, 50)
    out["MA200"] = _sma(close, 200)
    out["EMA12"] = _ema(close, 12)
    out["EMA26"] = _ema(close, 26)

    out["RSI"] = _rsi(close, window=14)

    stoch_k, stoch_d = _stochastic(high, low, close)
    out["STOCH_K"] = stoch_k
    out["STOCH_D"] = stoch_d

    out["ROC"] = _roc(close)
    out["WILLR"] = _williams_r(high, low, close)

    macd_line, signal_line, histogram = _macd(close)
    out["MACD"] = macd_line
    out["MACD_S"] = signal_line
    out["MACD_H"] = histogram

    bb_upper, bb_mid, bb_lower, bb_width = _bollinger_bands(close)
    out["BB_U"] = bb_upper
    out["BB_M"] = bb_mid
    out["BB_L"] = bb_lower
    out["BB_W"] = bb_width
    out["ATR"] = _atr(high, low, close)

    out["OBV"] = _obv(close, volume)
    out["VWAP"] = _vwap20(high, low, close, volume)

    adx, plus_di, minus_di = _adx(high, low, close)
    out["ADX"] = adx
    out["PLUS_DI"] = plus_di
    out["MINUS_DI"] = minus_di
    out["CCI"] = _cci(high, low, close)
    return out


def add_ml_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Features known at the close of day t. Target is the *next* session close.
    Same-day OHLC is valid here because the forecast is for t+1, after t has closed.
    """
    out = data.copy()
    out["Price_Range"] = out["High"] - out["Low"]
    out["Price_Change"] = out["Close"] - out["Open"]
    out["Ret_1"] = out["Close"].pct_change()
    vol_ma = out["Volume"].rolling(20).mean()
    out["Vol_Ratio"] = out["Volume"] / vol_ma.replace(0, np.nan)
    out["Target_Close"] = out["Close"].shift(-1)
    out["Target_Return"] = out["Close"].shift(-1) / out["Close"].replace(0, np.nan) - 1
    return out
