"""
Data fetching utilities using yfinance.
"""

import streamlit as st
import yfinance as yf
import pandas as pd


@st.cache_data(show_spinner=False)
def load_data(ticker: str, start, end) -> pd.DataFrame:
    """Download OHLCV data for *ticker* between *start* and *end*."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


@st.cache_data(show_spinner=False)
def load_info(ticker: str) -> dict:
    """Fetch company metadata (name, sector, industry …)."""
    try:
        info = yf.Ticker(ticker).info
        return info
    except:
        return {}
