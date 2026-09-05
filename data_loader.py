"""
Cached market fetch. Primary sources: NSE India + Stooq. Yahoo last.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import CACHE_TTL_SECONDS
from market_data import fetch_market, heatmap_returns


def make_demo_ohlcv(start, end, seed: int = 42) -> pd.DataFrame:
    """Deterministic daily OHLCV so the desk still runs with no network."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
        idx = pd.bdate_range("2021-01-04", periods=600)
    else:
        idx = pd.bdate_range(start_ts, end_ts)
        if len(idx) < 80:
            idx = pd.bdate_range(end_ts - pd.Timedelta(days=900), end_ts)
    n = len(idx)
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.00035, 0.012, n)
    close = 100.0 * np.cumprod(1 + rets)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0, 0.01, n))
    volume = rng.integers(800_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_market(ticker: str, start, end) -> dict:
    """Return ohlcv + source metadata. Cached 5 minutes."""
    ticker = (ticker or "").strip()
    empty = {
        "ohlcv": pd.DataFrame(),
        "source": "none",
        "source_id": "none",
        "info": {},
        "quote": {},
        "note": "",
        "live": False,
        "fetched_at": "",
    }
    if not ticker:
        return empty
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
        return empty
    result = fetch_market(ticker, start_ts, end_ts)
    # cache_data needs serialisable values; DataFrame is fine.
    return result


def load_data(ticker: str, start, end) -> pd.DataFrame:
    return load_market(ticker, start, end).get("ohlcv", pd.DataFrame())


def load_info(ticker: str) -> dict:
    # Kept for callers; prefer load_market()['info'].
    from config import COMPANY_META, TICKER_NAMES

    meta = COMPANY_META.get(ticker) or {}
    return {
        "longName": meta.get("longName") or TICKER_NAMES.get(ticker, ticker),
        "sector": meta.get("sector") or "N/A",
        "industry": meta.get("industry") or "N/A",
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_heatmap_returns(tickers: tuple[str, ...]) -> pd.DataFrame:
    tickers = tuple(t for t in tickers if t)
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "Change", "Lookback"])
    try:
        return heatmap_returns(tickers)
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Change", "Lookback"])
