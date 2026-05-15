"""
Application constants and stock ticker lists.
"""

# ─────────────────────────────────────────────
# STOCK TICKER LISTS
# ─────────────────────────────────────────────
INDIA_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS",
    "WIPRO.NS", "BAJFINANCE.NS", "ADANIENT.NS", "TATAMOTORS.NS",
]

GLOBAL_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN",
    "META", "NVDA", "NFLX", "AMD", "BABA",
]

ALL_TICKERS = {
    "RELIANCE.NS": "Reliance", "TCS.NS": "TCS", "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank", "SBIN.NS": "SBI", "ICICIBANK.NS": "ICICI",
    "WIPRO.NS": "Wipro", "TATAMOTORS.NS": "Tata Motors",
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "TSLA": "Tesla", "AMZN": "Amazon", "META": "Meta", "NVDA": "NVIDIA",
    "NFLX": "Netflix", "AMD": "AMD",
}

# ─────────────────────────────────────────────
# ML FEATURES
# ─────────────────────────────────────────────
FEATURES = [
    "Prev_Close", "Prev_Open", "Prev_High", "Prev_Low", "Prev_Volume",
    "Price_Range", "Price_Change", "MA20_Lag", "RSI_Lag", "MACD_Lag", "ATR_Lag",
]
