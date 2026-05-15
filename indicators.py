"""
Technical indicator computation using raw numpy and pandas only.
"""

import numpy as np
import pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d_window).mean()
    return stoch_k, stoch_d


def _roc(series: pd.Series, window: int = 12) -> pd.Series:
    return ((series - series.shift(window)) / series.shift(window).replace(0, np.nan)) * 100


def _williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


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
    std = series.rolling(window).std()
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
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = _atr(high, low, close, window)

    smoothed_plus_dm = plus_dm.ewm(alpha=1 / window, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100 * smoothed_plus_dm / atr.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_dm / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False).mean()


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


# ── Main Functions ────────────────────────────────────────────────────────────

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators using raw numpy and pandas,
    then drop rows with NaN values introduced by rolling windows.
    """
    close = data["Close"].squeeze()
    high = data["High"].squeeze()
    low = data["Low"].squeeze()
    volume = data["Volume"].squeeze()

    # ── Trend (Moving Averages) ──────────────────────────────────────────────
    data["MA20"] = _sma(close, 20)
    data["MA50"] = _sma(close, 50)
    data["MA200"] = _sma(close, 200)
    data["EMA12"] = _ema(close, 12)
    data["EMA26"] = _ema(close, 26)

    # ── Momentum ─────────────────────────────────────────────────────────────
    data["RSI"] = _rsi(close, window=14)

    stoch_k, stoch_d = _stochastic(high, low, close)
    data["STOCH_K"] = stoch_k
    data["STOCH_D"] = stoch_d

    data["ROC"] = _roc(close)
    data["WILLR"] = _williams_r(high, low, close)

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_line, signal_line, histogram = _macd(close)
    data["MACD"] = macd_line
    data["MACD_S"] = signal_line
    data["MACD_H"] = histogram

    # ── Volatility ───────────────────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower, bb_width = _bollinger_bands(close)
    data["BB_U"] = bb_upper
    data["BB_M"] = bb_mid
    data["BB_L"] = bb_lower
    data["BB_W"] = bb_width
    data["ATR"] = _atr(high, low, close)

    # ── Volume ───────────────────────────────────────────────────────────────
    data["OBV"] = _obv(close, volume)
    data["VWAP"] = _vwap(high, low, close, volume)

    # ── Trend Strength ───────────────────────────────────────────────────────
    data["ADX"] = _adx(high, low, close)
    data["CCI"] = _cci(high, low, close)

    data.dropna(inplace=True)
    return data


def add_ml_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged and engineered features used by the ML models.
    """
    # ── Lagged Price & Volume Columns ────────────────────────────────────────
    data["Prev_Close"] = data["Close"].shift(1)
    data["Prev_Open"] = data["Open"].shift(1)
    data["Prev_High"] = data["High"].shift(1)
    data["Prev_Low"] = data["Low"].shift(1)
    data["Prev_Volume"] = data["Volume"].shift(1)

    # ── Derived Price Features ────────────────────────────────────────────────
    data["Price_Range"] = data["High"] - data["Low"]
    data["Price_Change"] = data["Close"] - data["Open"]

    # ── Lagged Indicator Columns ──────────────────────────────────────────────
    data["MA20_Lag"] = data["MA20"].shift(1)
    data["RSI_Lag"] = data["RSI"].shift(1)
    data["MACD_Lag"] = data["MACD"].shift(1)
    data["ATR_Lag"] = data["ATR"].shift(1)

    return data