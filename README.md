<div align="center">

# 📈 AI Stock Dashboard
**Educational daily-bar analysis, causal next-session forecasts, and a simulated paper broker**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](LICENSE)

*India: NSE official traded bars. Global: Stooq EOD. Yahoo only if those fail. Not a live broker.*

[**Explore the Repository**](https://github.com/KrishnansuMohapatra/stock_knn)

</div>

---

## What this is

A Streamlit study desk on **free exchange data** (no paid key):

| Market | Primary source | What you get |
|---|---|---|
| India (`*.NS`) | **NSE India official** JSON | Unadjusted traded OHLC + last traded price |
| US / global | **Stooq** EOD CSV | Exchange session bars |
| Fallback only | Yahoo Finance | Unofficial delayed scrape — labelled in the UI |

Yahoo is **not** treated as real exchange data. If NSE and Stooq fail, the app says so.

- Candles with MA20 / MA50 / MA200 and Bollinger Bands
- Composite study signal (RSI, MACD *crossovers*, signed ADX, Stochastic, CCI, Williams %R, OBV)
- **Next-session close** models that only use information known at **today’s close**, plus a **persistence baseline** (`tomorrow ≈ today`)
- Simulated **paper ticket** (INR + USD cash, discount-broker fees, 1.5×ATR stop, 2R target)
- **Backtest** of the signal with **next-open fills**, slippage, fees, and buy-and-hold comparison

There is **no paid vendor and no brokerage API**. Paper trades never leave this browser session.

---

## What this is not

- Not real-time (daily bars, Yahoo delay, 1-hour cache)
- Not a live order gateway
- Not financial advice
- ML will often **lose to persistence**. That is shown on purpose.

---

## Signal map

| Score | Label |
|:---:|:---|
| **+35 to +100** | STRONG BUY |
| **+15 to +34** | BUY |
| **−14 to +14** | HOLD / NEUTRAL |
| **−34 to −15** | SELL |
| **−100 to −35** | STRONG SELL |

MACD points fire on an actual cross (previous bar vs this bar), not on every bar the histogram is positive. ADX is signed by +DI vs −DI.

---

## Quick start

### `uv`

```bash
git clone https://github.com/KrishnansuMohapatra/stock_knn.git
cd stock_knn
uv sync
uv run streamlit run app.py
```

### `pip`

```bash
git clone https://github.com/KrishnansuMohapatra/stock_knn.git
cd stock_knn
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

### Tests

```bash
python -m pytest tests -q
```

---

## Usage

1. Pick **India**, **Global**, or a **custom** ticker (`AAPL`, `RELIANCE.NS`, `BTC-USD`).
2. Keep at least ~60 trading days; MA200 needs ~200.
3. Read the study signal, then size a **paper** BUY/SELL. Fills are last close + slippage.
4. The backtest is the honest path: decisions at close, fills at the **next open**.
5. If no ML model beats **Persistence**, ignore the price forecast.

---

## Project structure

```text
stock_knn/
├── app.py              # Streamlit UI
├── config.py           # Tickers, features, paper defaults
├── utils.py            # Money / currency / sanitise
├── styles.py           # CSS
├── data_loader.py      # Yahoo + TTL cache + batch heatmap
├── indicators.py       # TA (pandas / numpy)
├── signals.py          # Composite score
├── ml_models.py        # Next-close models + persistence
├── broker.py           # Fees, sizing, paper blotter
├── backtest.py         # Next-open simulator
├── charts.py           # Plotly
├── report.py           # Narrative HTML
├── tests/              # Leakage, signals, fees
└── requirements.txt
```

---

## Troubleshooting

**No bars / demo series**
- India: use `RELIANCE.NS` (NSE). Confirm the symbol on [nseindia.com](https://www.nseindia.com).
- US: Stooq needs a plain ticker (`AAPL`), not `AAPL.NS`.
- If the UI says Yahoo fallback, the exchange feeds failed (network/SSL/rate-limit).

**Need at least 60 trading days**
- Widen the date range. Indicators are not dropped from the chart; MA200 simply stays blank until it exists.

**Heatmap missing names**
- Yahoo rate-limits. The heatmap is cached for an hour and skipped per-ticker on failure.

---

## Disclaimer

> For education and research only. Algorithmic signals and ML forecasts can be wrong. Simulated fills are not executions. Do your own research before any real-money decision.

---

<div align="center">
  <p>Built with Streamlit · yfinance · scikit-learn · Plotly</p>
</div>
