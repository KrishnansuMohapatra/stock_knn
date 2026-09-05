"""
Application constants, ticker universe, ML feature list, broker defaults.
"""

from __future__ import annotations

# ─────────────────────────────────────────────
# STOCK TICKER LISTS + STATIC META
# (live NSE quote overwrites name/sector when available)
# ─────────────────────────────────────────────
TICKER_NAMES = {
    "RELIANCE.NS": "Reliance",
    "TCS.NS": "TCS",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI",
    "SBIN.NS": "SBI",
    "ITC.NS": "ITC",
    "LT.NS": "L&T",
    "WIPRO.NS": "Wipro",
    "BAJFINANCE.NS": "Bajaj Fin.",
    "ADANIENT.NS": "Adani Ent.",
    "TATAMOTORS.NS": "Tata Motors",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "TSLA": "Tesla",
    "AMZN": "Amazon",
    "META": "Meta",
    "NVDA": "NVIDIA",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "BABA": "Alibaba",
}

COMPANY_META = {
    "RELIANCE.NS": {"longName": "Reliance Industries Limited", "sector": "Energy", "industry": "Oil & Gas Refining"},
    "TCS.NS": {"longName": "Tata Consultancy Services", "sector": "Information Technology", "industry": "IT Services"},
    "INFY.NS": {"longName": "Infosys Limited", "sector": "Information Technology", "industry": "IT Services"},
    "HDFCBANK.NS": {"longName": "HDFC Bank Limited", "sector": "Financials", "industry": "Private Bank"},
    "ICICIBANK.NS": {"longName": "ICICI Bank Limited", "sector": "Financials", "industry": "Private Bank"},
    "SBIN.NS": {"longName": "State Bank of India", "sector": "Financials", "industry": "Public Bank"},
    "ITC.NS": {"longName": "ITC Limited", "sector": "Consumer Staples", "industry": "FMCG"},
    "LT.NS": {"longName": "Larsen & Toubro Limited", "sector": "Industrials", "industry": "Engineering & Construction"},
    "WIPRO.NS": {"longName": "Wipro Limited", "sector": "Information Technology", "industry": "IT Services"},
    "BAJFINANCE.NS": {"longName": "Bajaj Finance Limited", "sector": "Financials", "industry": "NBFC"},
    "ADANIENT.NS": {"longName": "Adani Enterprises Limited", "sector": "Industrials", "industry": "Conglomerate"},
    "TATAMOTORS.NS": {"longName": "Tata Motors Limited", "sector": "Consumer Discretionary", "industry": "Automobiles"},
    "AAPL": {"longName": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics"},
    "MSFT": {"longName": "Microsoft Corporation", "sector": "Technology", "industry": "Software"},
    "GOOGL": {"longName": "Alphabet Inc.", "sector": "Communication Services", "industry": "Internet"},
    "TSLA": {"longName": "Tesla, Inc.", "sector": "Consumer Discretionary", "industry": "Automobiles"},
    "AMZN": {"longName": "Amazon.com, Inc.", "sector": "Consumer Discretionary", "industry": "E-commerce"},
    "META": {"longName": "Meta Platforms, Inc.", "sector": "Communication Services", "industry": "Social Media"},
    "NVDA": {"longName": "NVIDIA Corporation", "sector": "Technology", "industry": "Semiconductors"},
    "NFLX": {"longName": "Netflix, Inc.", "sector": "Communication Services", "industry": "Streaming"},
    "AMD": {"longName": "Advanced Micro Devices, Inc.", "sector": "Technology", "industry": "Semiconductors"},
    "BABA": {"longName": "Alibaba Group Holding Limited", "sector": "Consumer Discretionary", "industry": "E-commerce"},
}

INDIA_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS",
    "WIPRO.NS", "BAJFINANCE.NS", "ADANIENT.NS", "TATAMOTORS.NS",
]

GLOBAL_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN",
    "META", "NVDA", "NFLX", "AMD", "BABA",
]

ALL_TICKERS = {t: TICKER_NAMES.get(t, t) for t in INDIA_STOCKS + GLOBAL_STOCKS}

# ─────────────────────────────────────────────
# ML FEATURES — all known at today's close.
# Target is next session's close (see ml_models.py).
# ─────────────────────────────────────────────
FEATURES = [
    "Close", "Open", "High", "Low", "Volume",
    "Price_Range", "Price_Change", "Ret_1",
    "MA20", "RSI", "MACD", "ATR", "Vol_Ratio",
]

# ─────────────────────────────────────────────
# DATA / MODEL GUARDS
# ─────────────────────────────────────────────
MIN_BARS = 60
MIN_TRAIN_ROWS = 40
CACHE_TTL_SECONDS = 300  # 5 min so NSE LTP can refresh

# ─────────────────────────────────────────────
# PAPER BROKER DEFAULTS (free, simulated)
# ─────────────────────────────────────────────
DEFAULT_CASH_INR = 100_000.0
DEFAULT_CASH_USD = 10_000.0
DEFAULT_RISK_PCT = 0.01
ATR_STOP_MULT = 1.5
TAKE_PROFIT_R = 2.0
MAX_NOTIONAL_PCT = 0.95
