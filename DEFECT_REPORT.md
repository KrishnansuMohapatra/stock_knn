# Defect Report — AI Stock Dashboard (`stock_knn`)

> **Status (2026-09-05):** Critical/high items (ML leakage, MACD/ADX/Stochastic, crashes, heatmap, paper broker) were addressed in the follow-up implementation. This document remains the original audit of `e6cdf62`.

| Field | Value |
|---|---|
| **Project** | AI Stock Dashboard (Streamlit + yfinance + scikit-learn) |
| **Scope** | Full source tree (`app.py`, `ml_models.py`, `indicators.py`, `signals.py`, `charts.py`, `report.py`, `data_loader.py`, `config.py`, `styles.py`, packaging & docs) |
| **Method** | Static code review of every module, cross-check against README claims, and inspection of bundled screenshots (Reliance / AAPL, 15 May 2026) |
| **Date** | 5 Sep 2026 |
| **Code revision** | `e6cdf62` (`docs: make README look fantastic`) |

---

## 1. Executive summary

This is a small but ambitious dashboard: technical indicators, a composite trading signal, four regression models, a narrative report, and a market heatmap. The **UI composition is coherent** and the indicator library is reasonably complete. However, the **machine-learning “next-day forecast” is methodologically invalid**, several signal rules are wrong or misleading, short date ranges and failed network calls can **crash the app**, and a handful of UI/docs bugs contradict what the product claims to do.

The headline number in the screenshots — Ridge **MAPE ≈ 0.46%** on Reliance — is not evidence of forecasting skill. It is the expected result of **same-day target leakage**.

| Severity | Count | What it means |
|---|---:|---|
| Critical | 3 | Wrong answers presented as trading/ML output |
| High | 9 | Crashes, silently wrong analytics, or severe UX breakage |
| Medium | 12 | Incorrect edge behaviour, performance, security, or labelling |
| Low / Hygiene | 11 | Docs drift, unused code, missing tests |

**Do not treat the current Buy/Sell badge or next-day price as decision-grade.** Fix the ML target/features and the signal-engine bugs before any further model work.

---

## 2. Architecture snapshot

```
app.py  ──► data_loader.load_data / load_info     (yfinance, cached)
        ──► indicators.add_technical_indicators   (MA200 dropna)
        ──► signals.compute_signal_score
        ──► indicators.add_ml_features            (lags + same-day features)
        ──► ml_models.train_models / predict_next_day
        ──► charts.* / report.generate_report
        ──► charts.build_heatmap                  (uncached, 17 extra downloads)
```

There are **no tests, no CI, no LICENSE file**, and **no error handling** around the ML or heatmap paths.

---

## 3. Critical defects

### C1. Same-day feature leakage — models are not forecasting the next close

**Where:** `indicators.py` (`add_ml_features`), `config.py` (`FEATURES`), `ml_models.py` (`train_models`, `predict_next_day`)

The stated product behaviour (README + UI) is:

> Predict **next-day** closing prices using an ensemble of 4 regression models.

What the code actually does:

```python
# indicators.py — NOT lagged
data["Price_Range"]  = data["High"] - data["Low"]     # day's range
data["Price_Change"] = data["Close"] - data["Open"]   # contains Close_t

# ml_models.py — target is the SAME day's close
y = data_ml["Close"]
```

| Feature | Known at close of day *t−1*? | Used to predict `Close_t`? |
|---|---|---|
| `Prev_Close`, `Prev_Open`, `Prev_High`, `Prev_Low`, `Prev_Volume` | Yes | Yes (valid) |
| `MA20_Lag`, `RSI_Lag`, `MACD_Lag`, `ATR_Lag` | Yes | Yes (valid) |
| **`Price_Range` = High_t − Low_t** | **No** | **Yes — leak** |
| **`Price_Change` = Close_t − Open_t** | **No (contains the target)** | **Yes — leak** |

`Price_Change` is a linear transform of the target (`Close = Open + Price_Change`). Even without `Open` as an explicit column, tree and linear models can reconstruct close from this plus the lagged OHLC. That is why the screenshot shows:

| Model | MAPE % (Reliance screenshot) | Interpretation |
|---|---:|---|
| Ridge | **0.46** | Typical of reconstruction, not forecast |
| Gradient Boosting | 0.71 | Same |
| Random Forest | 0.84 | Same |
| KNN | 2.40 | Weaker interpolator, still not a true forecast |

A genuine next-day close forecast for a large-cap name usually has daily MAPE on the order of **1–3%+**, and should be compared to a **persistence baseline** (`ŷ_{t+1} = Close_t`). That baseline is never computed.

**Inference is also not next-day:**

```python
latest = results["X"].tail(1)          # last row = today's leaked features
preds[name] = model.predict(...)       # reconstructs today's close
```

The card labelled **“Next-Day Price Forecast”** is a re-prediction of the last bar.

**Fix:**

1. Target `y = Close.shift(-1)` (or equivalently shift all features by +1 and predict today’s close from yesterday).
2. Lag `Price_Range` / `Price_Change` (or drop them).
3. Drop the last row (unknown future target) before training; build the inference vector only from information available at the last close.
4. Report a persistence / naïve baseline next to RMSE/MAPE.

---

### C2. “Next-day” evaluation is in-sample reconstruction on a chronological split of the leaked task

**Where:** `ml_models.py` lines 21–24, `charts.py` `build_prediction_chart`

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
```

The chronological split is the right *shape* for time series, but because of C1 the test set is still “given today’s range and today’s `Close−Open`, guess today’s `Close`”. The Plotly chart (`y=y_test.values` with no dates) then sells this as model skill.

Ridge/GB lines in the screenshot hug Actual almost perfectly — the visual signature of leakage, not of a working predictor.

**Fix:** After C1, keep `shuffle=False`, add a gap (purge) between train and test, and plot against `y_test.index` so the user can see the hold-out window.

---

### C3. Signal engine mis-labels states as events and can score a downtrend as bullish

**Where:** `signals.py`

Three independent logic errors stack into a “composite AI score” that does not mean what the README table says.

#### C3a. MACD “crossover” is not a crossover

```python
if macd_val > macd_sig and macd_hist > 0:
    signals["MACD"] = ("Bullish Crossover", "bullish", +20)
```

`MACD_H = MACD − Signal`, so `macd_val > macd_sig` **iff** `macd_hist > 0`. The second clause is redundant. This fires on **every bar** where MACD is above its signal — possibly hundreds of consecutive days — not on the bar where they cross. Users are shown “Bullish Crossover” as if a fresh event occurred. Worth **±20**, the largest single weight.

#### C3b. ADX always votes buy when the trend is strong

```python
if adx_val > 25:
    trend_str, adx_pts = "Strong Trend", +5
```

ADX measures **strength, not direction**. A strong *sell-off* still receives **+5 (bullish)** toward the composite score, and the row is hard-coded `tone="neutral"`. In a crash this systematically biases the badge away from STRONG SELL.

#### C3c. Stochastic overbought/oversold fall-through is inverted

```python
if stoch_k < 20 and stoch_k > stoch_d:       # oversold AND cross
    ...
elif stoch_k > 80 and stoch_k < stoch_d:     # overbought AND cross
    ...
elif stoch_k > stoch_d:
    signals["Stochastic"] = ("Bullish", ..., +5)
else:
    signals["Stochastic"] = ("Bearish", ..., -5)
```

- Oversold but `%K ≤ %D` → labelled **Bearish (−5)** instead of oversold.
- Overbought but `%K ≥ %D` → labelled **Bullish (+5)** instead of overbought.

The AAPL screenshot shows the cross branch working; the fall-through branch is the broken one.

**Fix:** Detect crossovers with previous-bar comparison; sign ADX by `+DI/−DI` or by price vs MA; treat oversold/overbought as first-class states even without a cross.

---

## 4. High-severity defects

### H1. Unchecking “Show MACD” also hides the RSI chart

**Where:** `app.py` ~171–177

```python
if show_macd:
    col_macd, col_rsi = st.columns(2)
    ...
    st.plotly_chart(build_rsi_chart(...))
```

There is a sidebar checkbox for MACD and none for RSI, yet RSI is gated on `show_macd`. Confirmed by control flow; easy one-line fix (always show RSI, or add its own checkbox).

---

### H2. Short / valid date ranges crash after indicators are computed

**Where:** `app.py` (`prev()`), `signals.py` (`_prev`, OBV `iloc[-5]`), `report.py` (trend windows)

`add_technical_indicators` does `data.dropna()`, which **drops the first ~199 rows** because of `MA200`. Remaining length can still be small.

Then:

| Call | Minimum rows required | If shorter |
|---|---:|---|
| `prev("Close")` / `_prev(..., "MACD_H")` | 2 | `IndexError` |
| OBV slope `iloc[-5]` | 5 | `IndexError` |
| `data["MA50"].iloc[-50]` | 50 | `IndexError` |
| `data["MA20"].iloc[-20]` | 20 | `IndexError` |
| `data["MA200"].iloc[-60]` | 60 | `IndexError` |

README says “ensure at least 200 trading days” but the app does **not** check. A range of ~210 trading days produces a non-empty frame after `dropna` and then blows up in the report/signal section — a worse UX than the empty-data error.

**Fix:** After indicators, if `len(data) < 200` (or a documented minimum), `st.error` + `st.stop()`. Guard every `iloc[-n]`.

---

### H3. “No data found” is the only error for “not enough history for MA200”

**Where:** `app.py` 59–61, `indicators.py` 219

Selecting 3 months of data downloads successfully, then `dropna()` after MA200 empties the frame, then the user sees:

> ❌ No data found. Check ticker / date range.

That message is wrong (the ticker is fine; the window is too short). It also fires for delisted tickers, holidays-only ranges, inverted start/end, and Yahoo failures — one bucket for four causes.

---

### H4. Heatmap: uncached 17-ticker download on every rerun; empty result crashes

**Where:** `charts.py` `build_heatmap`

1. **No `@st.cache_data`.** Every widget change (KNN slider, RSI limits, Bollinger checkbox) re-downloads ~17 symbols sequentially.
2. **Bare `except: pass`** swallows rate-limits and TLS errors.
3. If every download fails:

```python
heat_df = pd.DataFrame(heat)          # no columns
heat_df["Abs"] = heat_df["Change"].abs()   # KeyError: 'Change'
```

4. `period="5d"` then `(close[-1] − close[0]) / close[0]` over **5 bars is a 4-interval return**. Yahoo’s own “5D” uses close vs close from 5 sessions ago (6 prints). Off-by-one vs the section title **“Market Heatmap (5-Day Return)”**.
5. Should be a single `yf.download(list(ALL_TICKERS), period="5d")`, not a Python loop.

---

### H5. Heatmap universe is missing tickers the rest of the app advertises

**Where:** `config.py`

| List | Tickers |
|---|---|
| `INDIA_STOCKS` extra | `ITC.NS`, `LT.NS`, `BAJFINANCE.NS`, `ADANIENT.NS` |
| `GLOBAL_STOCKS` extra | `BABA` |
| Missing from `ALL_TICKERS` | **all five above** |

The heatmap is not a view of the selected market; it is a stale, incomplete hard-coded dict.

---

### H6. Long-term trend uses a 60-day lookback labelled as MA200

**Where:** `report.py`

```python
trend_20  = ... data["MA20"].iloc[-1]  > data["MA20"].iloc[-20]   # consistent
trend_50  = ... data["MA50"].iloc[-1]  > data["MA50"].iloc[-50]   # consistent
trend_200 = ... data["MA200"].iloc[-1] > data["MA200"].iloc[-60]  # BUG
```

The report prints **“Long-term (MA200): Uptrend/Downtrend”** based on 60 sessions of MA200 slope. That can disagree with the actual 200-day direction and feeds the “all three timeframes align” sentence used in the overall assessment.

---

### H7. RSI is NaN (and the row is dropped) on zero-loss / zero-gain bars

**Where:** `indicators.py` `_rsi`

```python
rs = avg_gain / avg_loss.replace(0, np.nan)
return 100 - (100 / (1 + rs))
```

Wilder RSI definition:

- `avg_loss == 0` and `avg_gain > 0` → **RSI = 100**
- `avg_gain == 0` and `avg_loss > 0` → **RSI = 0**
- both 0 → typically hold previous / 50

Here both become **NaN**, and `data.dropna()` then **deletes those trading days** from charts, signals, and ML. A strong one-way move (or a zero-volume/unchanged bar) can punch holes in the series or even drop the **latest** session — the one the whole dashboard is about.

Same pattern: Stochastic / Williams `%R` divide by `(high − low)` after `replace(0, nan)`.

---

### H8. Top metrics overflow — price, 52W, volume are truncated in the layout

**Evidence:** bundled screenshot `screenshot_01_top_*.png`

Six `st.columns` on a wide layout still clip:

- Price `₹1,36…`
- 52W High `1,611…`
- 52W Low `1,290…`
- Volume `17,30…`

52W high/low also **omit the currency symbol** that Price uses, so Indian vs US values are harder to read even when not clipped.

The same screenshot’s analysis report shows the real numbers (`₹1,361.80`, 52W implied ~`1,611` / `1,290`) — the metrics row is a presentation bug, not a data bug.

---

### H9. No guard around `train_models` — KNN and split will throw

**Where:** `ml_models.py`, called from `app.py` with no `try`

Failure modes:

- After `dropna`, `len(X) < 2` → `train_test_split` fails.
- `len(X_train) < k` (sidebar allows `k=20`) → `KNeighborsRegressor` raises `Expected n_neighbors <= n_samples`.
- All-NaN feature column → `StandardScaler` raises.

Streamlit then shows a raw traceback instead of a user-facing message.

---

## 5. Medium-severity defects

### M1. yfinance `end` is exclusive — “today” usually excludes today’s bar

```python
end = st.sidebar.date_input("End Date", pd.to_datetime("today"))
df = yf.download(ticker, start=start, end=end, ...)
```

yfinance treats `end` as **exclusive**. Picking 15 May 2026 (the screenshot date) requests data **through 14 May**. Combined with no `ttl` on `st.cache_data`, the dashboard can sit on a stale last close across an entire server lifetime.

Also missing: `start < end` validation; inverted dates just look like C/H3.

---

### M2. Caches never expire

```python
@st.cache_data(show_spinner=False)
def load_data(...): ...
@st.cache_data(show_spinner=False)
def load_info(...): ...
```

No `ttl`. Overnight, the “price” and “AI signal” stay frozen until the Streamlit process restarts. `load_info` additionally caches the entire Yahoo `.info` blob (large, flaky, often rate-limited).

---

### M3. Currency heuristic is `.NS` only

```python
"₹" if ".NS" in ticker else "$"
```

Wrong for `RELIANCE.BO` (BSE), `VOD.L`, `7203.T`, `BTC-USD` (crypto is USD — OK by accident), `MC.PA`, etc. Custom-ticker mode advertises crypto and NSE in the README but only NSE gets rupees.

---

### M4. “52-week” high/low is `tail(252)` of the *post-MA200* series

If the user selects one year of data, ~199 rows are dropped, and “52W High” is the high of the remaining ~50 sessions. The metric name is then false. Should use a dedicated 52-week download (or `period="1y"`) independent of the chart window.

---

### M5. OBV text and colour disagree in the written report

**Where:** `report.py`

```html
<b class="{'bullish' if obv_val > 0 else 'bearish'}">
  {"rising (accumulation)" if obv_val > data["OBV"].iloc[-5] else "falling (distribution)"}
</b>
```

- Colour uses **sign of cumulative OBV** (almost always positive on a long bull market).
- Words use **5-day slope**.

A distribution day on Reliance is still painted green. Signals.py uses the slope for both label and points — the report should match that.

Related: OBV slope `== 0` is classified **“Falling – Selling Pressure (−10)”**.

---

### M6. Bollinger copy in the report is factually wrong

> “below lower band **(compressed)**”

Price below the lower band is an **extension / oversold** condition, not compression. Compression is a **narrow band width** (`BB_W`). The signal engine correctly treats below-lower as oversold (+12); the prose contradicts it.

---

### M7. XSS surface — custom ticker is injected into `unsafe_allow_html`

**Where:** `app.py` (`st.markdown(..., unsafe_allow_html=True)`), `report.py` interpolates `{ticker}` and `{company_name}` into HTML.

```python
ticker = st.sidebar.text_input("Enter Ticker", "AAPL").upper()
```

HTML tags are case-insensitive. A custom value such as `<IMG SRC=X ONERROR=...>` is uppercased and still parsed as HTML in the report block. Yahoo `longName` is also interpolated unsanitised.

This is a local dashboard, not a multi-tenant app, but Streamlit’s HTML sink is still an injection bug. Escape with `html.escape`.

---

### M8. ADX +DM / −DM equality case

**Where:** `indicators.py` `_adx`

`plus_dm` is zeroed **in place**, then `minus_dm` is compared to the *filtered* `plus_dm`. When raw `+DM == −DM` (both positive), standard Wilder sets **both to 0**; this implementation keeps `−DM`. Directional Index is then slightly biased. Seed/smoothing also skip the classic “SMA of first `window` bars, then Wilder” initialisation (RSI/ATR have the same approximation).

---

### M9. Bollinger width uses sample std (`ddof=1`)

Pandas `rolling(...).std()` defaults to `ddof=1`. TradingView / TA-Lib / most broker platforms use **population std (`ddof=0`)** for Bollinger Bands. Bands will not match the references traders compare against.

---

### M10. VWAP is a running lifetime VWAP, not session VWAP

```python
return (typical_price * volume).cumsum() / volume.cumsum()
```

This is VWAP from **the first row of the download** (2021-01-01 by default), not the current session / current day. The column is computed, then **never used** in signals, charts, or the report — dead, and wrong if anyone later wires it in.

`ROC`, `EMA12`, `EMA26`, `BB_W` are similarly computed and unused (except BB_W unused). Cost is small; confusion is not.

---

### M11. Prediction chart has no dates; MACD/RSI charts have no range slider isolation

```python
fig4.add_trace(go.Scatter(y=y_test.values, name="Actual", ...))
```

X-axis is `0, 50, 100, …` (visible in `screenshot_03_ml_*.png`). Users cannot tell which year the “test set” is.

RSI uses `fill="tozeroy"`, which in the screenshots renders as a heavy purple slab that obscures the 0–100 oscillator.

---

### M12. Retrain cost — every sidebar tweak fits RF + GB + Ridge + KNN

Changing **RSI overbought**, **Show Bollinger**, or **k** retrains `RandomForestRegressor(n_estimators=100, n_jobs=-1)` and GBR on the full history. Only `k` should retrain KNN. Combined with H4 (heatmap), a single slider drag is extremely expensive and Yahoo-hostile.

`n_jobs=-1` inside Streamlit also fights the app server’s own threads.

---

## 6. Low severity & project hygiene

| ID | Issue | Location |
|---|---|---|
| L1 | Footer claims **“Plotly · TA-Lib”** but TA-Lib is not a dependency and is not used | `app.py` last markdown |
| L2 | README MIT badge links to `LICENSE`; **no LICENSE file** | repo root |
| L3 | README: “**real-time**” analysis — data is **daily** yfinance bars | README, product copy |
| L4 | `matplotlib` is in `requirements.txt` / `pyproject.toml` but **never imported** | packaging |
| L5 | `uv.lock` is **gitignored** — `uv sync` is not reproducible | `.gitignore` |
| L6 | `ALL_TICKERS` / `INDIA_STOCKS` / `GLOBAL_STOCKS` triplicated instead of one source of truth | `config.py` |
| L7 | Bare `except:` in `load_info` and heatmap (hides `KeyboardInterrupt`-adjacent bugs, rate limits, typos) | `data_loader.py`, `charts.py` |
| L8 | `pandas` imported in `report.py` and `signals.py` but unused (signals: annotations only) | — |
| L9 | Ensemble is an **unweighted mean** of four models, including the worst (KNN). No weighting by MAPE | `ml_models.py` |
| L10 | Correlated oscillators (RSI, Stochastic, Williams %R, CCI) are summed as if independent — double/triple counting of the same overbought reading | `signals.py` |
| L11 | **Zero automated tests**, no `tests/`, no GitHub Actions, no pre-commit | repo |

---

## 7. Documentation vs product (traceability)

| Claim | Reality |
|---|---|
| “Next-day closing prices” | Predicts **same-day** close with leaked OHLC (C1) |
| “Real-time technical analysis” | End-of-day bars, exclusive `end`, cache with no TTL |
| “9+ technical indicators” aggregated | Several are unused (ROC, VWAP); ADX/MACD/Stoch rules are wrong (C3) |
| Score −100…+100 table | Theoretical raw sum ≈ −107…+110, then clamped; ADX can push the wrong way |
| “KNN neighbours slider … prediction changes in real time” | Also silently retrains RF/GB and redownloads the heatmap |
| Heatmap of “major Indian and US stocks” | Missing ITC, LT, Bajaj Finance, Adani, BABA |
| Built with TA-Lib | Not used |
| License: MIT | No license file |
| Troubleshooting: 200 trading days | App does not enforce it and can crash instead |

---

## 8. What is *not* broken (so effort is not wasted)

- Chronological `train_test_split(..., shuffle=False)` is the correct split *style*.
- `StandardScaler` is fit on train only (no scaler leak).
- MACD EMA (`adjust=False`) matches the common TradingView construction.
- Signal thresholds in `classify_signal` match the README table (`±15`, `±35`).
- Candlestick + MA20/50/200 + optional BB wiring is correct.
- Educational disclaimer exists in both README and the generated report.
- No API keys / secrets in the tree.

---

## 9. Recommended fix order

| Priority | Item | Effort | Risk if skipped |
|---|---|---|---|
| P0 | **C1 + C2** — lag all features, target `Close.shift(-1)`, add persistence baseline | S | Entire ML section is misleading |
| P0 | **C3** — MACD event vs state, ADX direction, Stochastic fall-through | S | Badge can say STRONG BUY/SELL for the wrong reason |
| P1 | **H2 + H3 + H9** — minimum-row checks, distinct error messages, try/except around ML | S | App crashes on normal user input |
| P1 | **H1** — stop gating RSI on `show_macd` | XS | Obvious UI bug |
| P1 | **H4 + H5 + M12** — cache heatmap, batch download, don’t retrain RF when `k` changes | M | Unusable latency / Yahoo bans |
| P2 | **H6, H7, M4, M5, M6** — report math and RSI zero-loss | S | Wrong narrative |
| P2 | **H8, M3, M11** — metrics layout, currency, chart dates | S | Polish |
| P3 | Tests for indicators vs a golden series, leakage unit test (`Price_Change` must not contain `Close_t`), signal-table fixtures | M | Regressions will return |
| P3 | LICENSE, drop unused matplotlib/TA-Lib claim, pin lockfile, `ttl=` on cache | XS | Hygiene |

### Suggested leakage unit test (should fail on current `main`)

```python
def test_ml_features_do_not_contain_same_day_close(df):
    out = add_ml_features(add_technical_indicators(df.copy()))
    # Price_Change must not be a function of today's Close
    assert "Price_Change" not in out.columns or out["Price_Change"].equals(
        out["Close"].shift(1) - out["Open"].shift(1)
    )
    y_next = out["Close"].shift(-1)
    assert not out["Price_Change"].corr(out["Close"]) > 0.95  # current code fails here
```

---

## 10. File-by-file defect index

| File | Defects |
|---|---|
| `ml_models.py` | C1, C2, H9, L9 |
| `indicators.py` | C1, H7, M8, M9, M10 |
| `signals.py` | C3a/b/c, L10, M5-related OBV zero |
| `app.py` | H1, H2, H3, H8, M1, M3, M7, M12, L1 |
| `charts.py` | H4, C2 (no dates), M11 |
| `report.py` | H6, M5, M6, H2 |
| `config.py` | C1 (feature list), H5, L6 |
| `data_loader.py` | M1, M2, L7 |
| `README.md` / packaging | L2–L5, section 7 mismatches |

---

## 11. Conclusion

The dashboard is a good **front-end shell** around a fragile analytics core. The two defects that actually change a user’s decision are:

1. **The ML block is not a forecast** (same-day leakage + last-row inference).
2. **The composite signal mis-states MACD, ADX, and Stochastic**, so the STRONG BUY / STRONG SELL card can be internally inconsistent with the indicators it claims to aggregate.

Everything else — crashes on short windows, heatmap behaviour, truncated metrics, docs drift — is real, but secondary. Fix P0 before adding more models or more tickers.
