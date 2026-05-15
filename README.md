# 📈 AI Stock Dashboard

An intelligent, real-time stock analysis dashboard built with **Streamlit** that combines **technical analysis**, **machine learning price prediction**, and **automated report generation** — all in a single interactive web app.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55+-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔍 What Is This?

This project is an **AI-powered stock market dashboard** that lets you:

- **Analyze any stock** from the Indian (NSE) or Global (US) markets — or type in any custom ticker.
- **View interactive candlestick charts** with moving averages and Bollinger Bands overlaid.
- **Get AI-generated Buy / Sell / Hold signals** based on a composite score from 9+ technical indicators (RSI, MACD, Stochastic, ADX, CCI, Williams %R, OBV, Bollinger Bands, and Moving Averages).
- **Predict next-day prices** using 4 ML models: KNN, Ridge Regression, Random Forest, and Gradient Boosting.
- **Read an auto-generated analysis report** covering price summary, trend analysis, momentum, volatility, volume, ML predictions, and overall assessment.
- **See a market heatmap** showing 5-day returns across major Indian and US stocks.

---

## 🧠 How It Works

### 1. Data Fetching
The app downloads historical OHLCV (Open, High, Low, Close, Volume) data from **Yahoo Finance** using the `yfinance` library. You choose the stock ticker and date range from the sidebar.

### 2. Technical Indicators
All indicators are computed using **raw NumPy and Pandas** (no external TA library needed):

| Category | Indicators |
|---|---|
| **Trend** | MA20, MA50, MA200, EMA12, EMA26 |
| **Momentum** | RSI (14), Stochastic K/D, ROC, Williams %R |
| **MACD** | MACD Line, Signal Line, Histogram |
| **Volatility** | Bollinger Bands (Upper/Mid/Lower/Width), ATR |
| **Volume** | OBV (On-Balance Volume), VWAP |
| **Trend Strength** | ADX, CCI |

### 3. AI Signal Engine
Each indicator "votes" with a weighted score. The votes are summed into a **composite score from -100 to +100**:

| Score Range | Signal |
|---|---|
| +35 to +100 | 🟢🟢 **STRONG BUY** |
| +15 to +34 | 🟢 **BUY** |
| -14 to +14 | 🟡 **HOLD / NEUTRAL** |
| -34 to -15 | 🔴 **SELL** |
| -100 to -35 | 🔴🔴 **STRONG SELL** |

### 4. Machine Learning Prediction
Four regression models are trained on engineered features (lagged prices, indicators, volume):

- **KNN Regressor** — neighbours-based, adjustable K from sidebar
- **Ridge Regression** — regularised linear model
- **Random Forest** — ensemble of decision trees
- **Gradient Boosting** — sequential boosting trees

The models are evaluated using RMSE and MAPE, and an **ensemble average** provides the final next-day price forecast.

### 5. Auto Analysis Report
A detailed HTML report is generated covering price position, trend alignment, momentum state, volatility regime, volume analysis, ML results, and an overall buy/sell assessment.

---

## 📁 Project Structure

```
stock_knn/
├── app.py              # Main Streamlit entry-point (orchestrator)
├── config.py           # Stock ticker lists, constants, ML feature names
├── styles.py           # Custom CSS for the dashboard
├── data_loader.py      # yFinance data fetching with Streamlit caching
├── indicators.py       # Technical indicator computation (raw numpy/pandas)
├── signals.py          # Composite signal scoring engine
├── ml_models.py        # ML model training, evaluation, prediction
├── charts.py           # Plotly chart builders
├── report.py           # Auto-generated HTML analysis report
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project metadata (uv/pip)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## 🚀 Setup & Installation

### Prerequisites
- **Python 3.11+** installed on your system
- **Internet connection** (to fetch stock data from Yahoo Finance)

### Option A: Using `pip`

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/stock_knn.git
cd stock_knn

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

### Option B: Using `uv` (faster)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/stock_knn.git
cd stock_knn

# 2. Install dependencies and create venv automatically
uv sync

# 3. Run the app
uv run streamlit run app.py
```

### After Running
The terminal will display a local URL like:
```
Local URL: http://localhost:8501
```
Open this URL in your browser to use the dashboard.

---

## 🛠️ Usage

1. **Choose a market** from the sidebar: 🇮🇳 India, 🌍 Global, or ⭐ Custom.
2. **Select a stock** or type any valid Yahoo Finance ticker (e.g., `RELIANCE.NS`, `AAPL`, `TSLA`).
3. **Adjust the date range** and **KNN neighbours** as needed.
4. **Tweak indicator settings** — RSI overbought/oversold thresholds, toggle Bollinger Bands and MACD charts.
5. **Scroll down** to see the AI signal, charts, ML predictions, auto report, and market heatmap.

---

## ⚠️ Troubleshooting: Data Not Loading

Sometimes stock data may fail to load. Here are common causes and fixes:

### ❌ "No data found. Check ticker / date range."

| Cause | Fix |
|---|---|
| **Wrong ticker symbol** | Verify the symbol on [finance.yahoo.com](https://finance.yahoo.com). Indian stocks need `.NS` suffix (e.g., `RELIANCE.NS`, not `RELIANCE`). |
| **Stock delisted** | Some tickers get delisted or renamed. For example, `TATAMOTORS.NS` may not work — try `TATAMTRDVS.NS` or search Yahoo Finance for the updated symbol. |
| **Date range too narrow** | The app needs at least 200+ trading days to compute MA200. Use a start date at least 1 year back. |
| **Weekend / holiday** | If start and end dates fall on non-trading days with no data in between, the download will be empty. |
| **No internet** | `yfinance` needs an active internet connection. Check your network. |

### ❌ Heatmap shows some stocks missing

The market heatmap tries to load data for all preset tickers. If a company has been delisted, renamed, or Yahoo Finance returns an error, that stock is **silently skipped** — the rest of the heatmap still works fine. This is expected behavior.

### ❌ "ModuleNotFoundError"

You haven't installed the dependencies. Run:
```bash
pip install -r requirements.txt
```

### ❌ Slow first load

The first run downloads data from Yahoo Finance and trains ML models — this can take 10–30 seconds. Subsequent runs are faster thanks to Streamlit's built-in caching (`@st.cache_data`).

### 💡 Tips for Custom Tickers

- **US stocks**: Use the plain symbol → `AAPL`, `TSLA`, `GOOGL`
- **Indian NSE stocks**: Append `.NS` → `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- **Indian BSE stocks**: Append `.BO` → `RELIANCE.BO`
- **Crypto**: Use Yahoo format → `BTC-USD`, `ETH-USD`
- **ETFs**: Plain symbols → `SPY`, `QQQ`, `NIFTYBEES.NS`

---

## 📊 Supported Pre-Loaded Stocks

### 🇮🇳 India (NSE)
`RELIANCE.NS` · `TCS.NS` · `INFY.NS` · `HDFCBANK.NS` · `ICICIBANK.NS` · `SBIN.NS` · `ITC.NS` · `LT.NS` · `WIPRO.NS` · `BAJFINANCE.NS` · `ADANIENT.NS` · `TATAMOTORS.NS`

### 🌍 Global (US)
`AAPL` · `MSFT` · `GOOGL` · `TSLA` · `AMZN` · `META` · `NVDA` · `NFLX` · `AMD` · `BABA`

---

## ⚖️ Disclaimer

> This tool is built for **educational and research purposes only**. It does **not** constitute financial advice. The AI signals and ML predictions are algorithmic outputs — they can be wrong. Always do your own research (DYOR) before making any investment decisions.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/awesome-thing`)
3. Commit your changes (`git commit -m "Add awesome thing"`)
4. Push to the branch (`git push origin feature/awesome-thing`)
5. Open a Pull Request

---

## 📜 License

This project is open source under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using Streamlit · yFinance · scikit-learn · Plotly
</p>
#   s t o c k _ k n n  
 