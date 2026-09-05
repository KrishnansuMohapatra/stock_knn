"""Unit tests for leakage, signals, fees, and backtest causality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import run_signal_backtest
from broker import (
    COST_PRESETS,
    empty_book,
    execute,
    fee_on_fill,
    position_size,
    slipped_price,
)
from charts import build_heatmap
from config import FEATURES, MIN_BARS
from data_loader import make_demo_ohlcv
from indicators import add_ml_features, add_technical_indicators, _rsi
from market_data import _normalize_ohlcv, is_india_ticker, nse_symbol, stooq_symbol
from ml_models import train_models
from signals import classify_signal, compute_signal_score
from utils import currency_for, money_symbol


def _ohlcv(n=400, start=100.0, seed=0, drift=0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, n)
    close = start * np.cumprod(1 + rets)
    open_ = np.concatenate([[start], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0, 0.008, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0, 0.008, n))
    vol = rng.integers(1_000_000, 4_000_000, n).astype(float)
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def test_rsi_monotone_up_is_100_not_nan():
    close = pd.Series(np.linspace(10, 50, 40))
    rsi = _rsi(close, 14)
    tail = rsi.iloc[20:]
    assert tail.notna().all()
    assert (tail > 90).all()


def test_rsi_flat_is_50():
    close = pd.Series([10.0] * 40)
    rsi = _rsi(close, 14)
    assert abs(float(rsi.iloc[-1]) - 50.0) < 1e-6


def test_ml_target_is_next_close_not_same_day():
    raw = _ohlcv()
    data = add_ml_features(add_technical_indicators(raw))
    valid = data.dropna(subset=FEATURES + ["Target_Close"])
    assert not np.allclose(valid["Target_Close"], valid["Close"])
    assert np.allclose(
        valid["Target_Close"].to_numpy(),
        data["Close"].shift(-1).loc[valid.index].to_numpy(),
        equal_nan=True,
    )
    # same-day range is allowed as a feature for t+1, but must not equal the target
    assert not np.allclose(valid["Price_Change"], valid["Target_Close"] - valid["Open"])


def test_train_models_beats_or_reports_persistence():
    raw = _ohlcv(n=500)
    data = add_ml_features(add_technical_indicators(raw))
    results = train_models(data, FEATURES, k=5)
    assert "error" not in results
    assert "Persistence" in results["perf_df"].index
    y_test = results["y_test"]
    close_test = results["close_test"]
    # persistence prediction is today's close, not today's target
    assert np.allclose(results["predictions"]["Persistence"], close_test.to_numpy())
    assert not np.allclose(y_test.to_numpy(), close_test.to_numpy())


def test_macd_crossover_is_event_not_state():
    n = 30
    df = pd.DataFrame({
        "Close": np.linspace(100, 110, n),
        "RSI": np.full(n, 50.0),
        "MACD": np.concatenate([np.full(15, -1.0), np.full(15, 1.0)]),
        "MACD_S": np.zeros(n),
        "MACD_H": np.concatenate([np.full(15, -1.0), np.full(15, 1.0)]),
        "MA20": np.linspace(100, 110, n),
        "MA50": np.linspace(99, 109, n),
        "MA200": np.linspace(90, 100, n),
        "BB_U": np.full(n, 120.0),
        "BB_L": np.full(n, 80.0),
        "BB_M": np.full(n, 100.0),
        "STOCH_K": np.full(n, 50.0),
        "STOCH_D": np.full(n, 50.0),
        "ADX": np.full(n, 15.0),
        "PLUS_DI": np.full(n, 20.0),
        "MINUS_DI": np.full(n, 20.0),
        "CCI": np.zeros(n),
        "WILLR": np.full(n, -50.0),
        "OBV": np.linspace(0, 100, n),
    })
    _, _ = compute_signal_score(df, 70, 30, i=14)
    sig_cross, _ = compute_signal_score(df, 70, 30, i=15)
    sig_after, _ = compute_signal_score(df, 70, 30, i=20)
    assert sig_cross["MACD"][0] == "Bullish Crossover"
    assert sig_after["MACD"][0] == "Above signal"


def test_adx_downtrend_is_bearish_points():
    n = 10
    df = pd.DataFrame({
        "Close": np.full(n, 50.0),
        "RSI": np.full(n, 50.0),
        "MACD": np.full(n, -0.2),
        "MACD_S": np.zeros(n),
        "MACD_H": np.full(n, -0.2),
        "MA20": np.full(n, 51.0),
        "MA50": np.full(n, 52.0),
        "MA200": np.full(n, 53.0),
        "BB_U": np.full(n, 60.0),
        "BB_L": np.full(n, 40.0),
        "BB_M": np.full(n, 50.0),
        "STOCH_K": np.full(n, 50.0),
        "STOCH_D": np.full(n, 50.0),
        "ADX": np.full(n, 30.0),
        "PLUS_DI": np.full(n, 10.0),
        "MINUS_DI": np.full(n, 25.0),
        "CCI": np.zeros(n),
        "WILLR": np.full(n, -50.0),
        "OBV": np.linspace(100, 0, n),
    })
    sig, _ = compute_signal_score(df, 70, 30, i=5)
    assert sig["ADX"][2] < 0
    assert "downtrend" in sig["ADX"][0].lower()


def test_stochastic_oversold_without_cross_is_bullish():
    n = 8
    df = pd.DataFrame({
        "Close": np.full(n, 50.0),
        "RSI": np.full(n, 50.0),
        "MACD": np.zeros(n),
        "MACD_S": np.zeros(n),
        "MACD_H": np.zeros(n),
        "MA20": np.full(n, 50.0),
        "MA50": np.full(n, 50.0),
        "MA200": np.full(n, 50.0),
        "BB_U": np.full(n, 60.0),
        "BB_L": np.full(n, 40.0),
        "BB_M": np.full(n, 50.0),
        "STOCH_K": np.full(n, 10.0),
        "STOCH_D": np.full(n, 15.0),  # K < D, oversold
        "ADX": np.full(n, 15.0),
        "PLUS_DI": np.full(n, 20.0),
        "MINUS_DI": np.full(n, 20.0),
        "CCI": np.zeros(n),
        "WILLR": np.full(n, -50.0),
        "OBV": np.full(n, 1.0),
    })
    sig, _ = compute_signal_score(df, 70, 30, i=5)
    assert sig["Stochastic"][2] > 0
    assert "Oversold" in sig["Stochastic"][0]


def test_classify_thresholds():
    assert classify_signal(35)[0] == "STRONG BUY"
    assert classify_signal(15)[0] == "BUY"
    assert classify_signal(0)[0] == "HOLD / NEUTRAL"
    assert classify_signal(-15)[0] == "SELL"
    assert classify_signal(-35)[0] == "STRONG SELL"


def test_india_sell_includes_stt():
    model = COST_PRESETS["india"]
    buy_fee = fee_on_fill(100_000, "BUY", model)
    sell_fee = fee_on_fill(100_000, "SELL", model)
    assert sell_fee > buy_fee
    assert pytest.approx(sell_fee - buy_fee, rel=1e-6) == 100_000 * model.stt_sell_pct


def test_slippage_hurts_the_trader():
    model = COST_PRESETS["us"]
    assert slipped_price(100, "BUY", model) > 100
    assert slipped_price(100, "SELL", model) < 100


def test_paper_buy_and_sell_cash():
    book = empty_book(cash_inr=100_000, cash_usd=10_000)
    book, msg = execute(book, "RELIANCE.NS", "BUY", 10, 1000.0)
    assert msg == "ok"
    assert "RELIANCE.NS" in book["positions"]
    assert book["cash"]["INR"] < 100_000
    book, msg = execute(book, "RELIANCE.NS", "SELL", 10, 1000.0)
    assert msg == "ok"
    assert "RELIANCE.NS" not in book["positions"]


def test_sell_without_position_rejected():
    book = empty_book()
    _, msg = execute(book, "AAPL", "SELL", 5, 100.0)
    assert msg != "ok"


def test_position_size_respects_cash():
    model = COST_PRESETS["us"]
    qty = position_size(price=100, stop=99, equity=10_000, cash=150, model=model, risk_pct=0.5)
    assert qty <= 1


def test_currency_helpers():
    assert currency_for("TCS.NS") == "INR"
    assert currency_for("RELIANCE.BO") == "INR"
    assert currency_for("AAPL") == "USD"
    assert money_symbol("INFY.NS") == "₹"
    assert money_symbol("MSFT") == "$"


def test_empty_heatmap_does_not_raise():
    fig = build_heatmap(pd.DataFrame(), {})
    assert fig is not None


def test_backtest_fills_use_next_open():
    raw = _ohlcv(n=300, seed=1)
    data = add_technical_indicators(raw)
    bt = run_signal_backtest(
        data, 70, 30, "AAPL",
        initial_capital=10_000,
        long_short=False,
        risk_pct=0.02,
    )
    assert "error" not in bt
    assert not bt["equity"].empty
    trades = bt["trades"]
    if trades is None or trades.empty:
        pytest.skip("no trades on this synthetic path")
    first = trades.iloc[0]
    day = first["date"]
    # fill must equal slipped next-open, not that day's close
    row = data.loc[day]
    assert first["fill"] != pytest.approx(float(row["Close"]), rel=1e-9)


def test_nse_and_stooq_symbol_mapping():
    assert nse_symbol("RELIANCE.NS") == "RELIANCE"
    assert nse_symbol("SBIN.BO") == "SBIN"
    assert is_india_ticker("TCS.NS")
    assert not is_india_ticker("AAPL")
    assert stooq_symbol("AAPL") == "aapl.us"
    assert stooq_symbol("RELIANCE.NS") == "reliance.in"
    assert stooq_symbol("VOD.L") == "vod.uk"
    assert stooq_symbol("BTC-USD") == "btc.v"


def test_normalize_ohlcv_from_nse_shaped_rows():
    raw = pd.DataFrame(
        {
            "CH_TIMESTAMP": ["2024-01-02", "2024-01-03"],
            "CH_OPENING_PRICE": [100, 101],
            "CH_TRADE_HIGH_PRICE": [102, 103],
            "CH_TRADE_LOW_PRICE": [99, 100],
            "CH_CLOSING_PRICE": [101, 102],
            "CH_TOT_TRADED_QTY": [1000, 1100],
        }
    )
    out = _normalize_ohlcv(raw)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out) == 2
    assert float(out["Close"].iloc[-1]) == 102


def test_demo_ohlcv_is_trainable():
    raw = make_demo_ohlcv("2021-01-01", "2024-12-31")
    assert len(raw) >= MIN_BARS
    data = add_ml_features(add_technical_indicators(raw))
    results = train_models(data, FEATURES, k=5)
    assert "error" not in results
    assert results["y_test"].shape[0] >= 5


def test_backtest_no_crash_on_short_history():
    raw = _ohlcv(n=80, seed=2)
    data = add_technical_indicators(raw)
    bt = run_signal_backtest(data, 70, 30, "AAPL", initial_capital=10_000)
    assert isinstance(bt, dict)
