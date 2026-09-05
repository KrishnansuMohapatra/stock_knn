"""
Free *exchange* data — not Yahoo as the primary source.

India  → NSE India official JSON (traded OHLC + live LTP). Unadjusted.
Global → Stooq EOD (exchange session bars).
Yahoo  → last-resort unofficial scrape only, labelled as such.

No API keys. No paid vendor.
"""

from __future__ import annotations

import io
import math
from datetime import datetime

import pandas as pd
import requests

from config import COMPANY_META, TICKER_NAMES
from utils import currency_for

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

NSE_HOME = "https://www.nseindia.com"
NSE_QUOTE = "https://www.nseindia.com/api/quote-equity"
NSE_HISTORY = "https://www.nseindia.com/api/historical/cm/equity"
NSE_HISTORY_OR = "https://www.nseindia.com/api/historicalOR/cm/equity"

_SESSION: requests.Session | None = None


def nse_symbol(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    for suffix in (".NS", ".BO", ".BSE"):
        if t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def is_india_ticker(ticker: str) -> bool:
    t = (ticker or "").upper()
    return t.endswith((".NS", ".BO", ".BSE")) or currency_for(t) == "INR"


def stooq_symbol(ticker: str) -> str:
    t = (ticker or "").strip().lower()
    if not t:
        return t
    if t.endswith(".ns") or t.endswith(".bo"):
        return t.split(".")[0] + ".in"
    if t.endswith(".bse"):
        return t[:-4] + ".in"
    if t.endswith(".l"):
        return t[:-2] + ".uk"
    if t.endswith(".t"):
        return t[:-2] + ".jp"
    if t.endswith(".hk"):
        return t[:-3] + ".hk"
    if t.endswith(".de"):
        return t[:-3] + ".de"
    if t.endswith("-usd"):
        return t.replace("-usd", "") + ".v"
    if "." not in t:
        return t + ".us"
    return t


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    out = df.copy()
    mapping = {}
    for c in out.columns:
        key = str(c).lower().replace(" ", "").replace("_", "")
        if key in {"open", "o", "chopeningprice"}:
            mapping[c] = "Open"
        elif key in {"high", "h", "chtradehighprice", "highprice"}:
            mapping[c] = "High"
        elif key in {"low", "l", "chtradelowprice", "lowprice"}:
            mapping[c] = "Low"
        elif key in {"close", "c", "chclosingprice", "ltp", "last", "chlasttradedprice"}:
            mapping[c] = "Close"
        elif key in {"volume", "vol", "chtottradedqty", "qty", "tottrdqty"}:
            mapping[c] = "Volume"
        elif key in {"date", "timestamp", "chtimestamp", "mdate"}:
            mapping[c] = "Date"
    out = out.rename(columns=mapping)
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"]).set_index("Date")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()]
    out.index = out.index.tz_localize(None) if getattr(out.index, "tz", None) else out.index
    out.index = out.index.normalize()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in out.columns for c in needed):
        return pd.DataFrame(columns=needed)
    out = out[needed].copy()
    for c in needed:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(how="any")
    out = out[out["Close"] > 0]
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _nse_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{NSE_HOME}/",
            "Connection": "keep-alive",
        }
    )
    try:
        s.get(NSE_HOME, timeout=15)
        s.get(f"{NSE_HOME}/option-chain", timeout=15)
    except Exception:
        pass
    _SESSION = s
    return s


def reset_nse_session() -> None:
    global _SESSION
    _SESSION = None


def _nse_get(url: str, params: dict | None = None) -> dict | list | None:
    s = _nse_session()
    try:
        r = s.get(url, params=params, timeout=20)
        if r.status_code in {401, 403}:
            reset_nse_session()
            s = _nse_session()
            r = s.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return None
        ctype = r.headers.get("content-type", "")
        if "json" not in ctype and not r.text.lstrip().startswith(("{", "[")):
            return None
        return r.json()
    except Exception:
        return None


def fetch_nse_quote(ticker: str) -> dict:
    """Official NSE last traded snapshot (LTP, day OHLC, % change)."""
    sym = nse_symbol(ticker)
    if not sym:
        return {}
    payload = _nse_get(NSE_QUOTE, {"symbol": sym})
    if not isinstance(payload, dict):
        return {}
    info = payload.get("info") or {}
    price = payload.get("priceInfo") or {}
    industry = payload.get("industryInfo") or {}
    trade = ((payload.get("securityWiseDP") or {}) if isinstance(payload.get("securityWiseDP"), dict) else {})
    md = payload.get("marketDeptOrderBook") or {}
    trade_info = (md.get("tradeInfo") or {}) if isinstance(md, dict) else {}
    hl = price.get("intraDayHighLow") or {}
    last = price.get("lastPrice")
    if last is None:
        return {}
    volume = (
        trade_info.get("totalTradedVolume")
        or trade.get("quantityTraded")
        or price.get("totalTradedVolume")
        or 0
    )
    return {
        "symbol": info.get("symbol") or sym,
        "longName": info.get("companyName") or TICKER_NAMES.get(ticker, ticker),
        "sector": industry.get("sector") or industry.get("macro") or "N/A",
        "industry": industry.get("industry") or info.get("industry") or "N/A",
        "lastPrice": float(last),
        "change": float(price.get("change") or 0),
        "pChange": float(price.get("pChange") or 0),
        "open": float(price.get("open") or last),
        "high": float(hl.get("max") or price.get("weekHigh") or last),
        "low": float(hl.get("min") or price.get("weekLow") or last),
        "previousClose": float(price.get("previousClose") or last),
        "volume": float(volume or 0),
        "source": "NSE India (official LTP)",
    }


def _history_rows_to_df(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    recs = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        dt = row.get("CH_TIMESTAMP") or row.get("mTIMESTAMP") or row.get("CH_DATE") or row.get("DATE")
        recs.append(
            {
                "Date": dt,
                "Open": row.get("CH_OPENING_PRICE") or row.get("CH_OPEN_PRICE") or row.get("OPEN"),
                "High": row.get("CH_TRADE_HIGH_PRICE") or row.get("CH_HIGH_PRICE") or row.get("HIGH"),
                "Low": row.get("CH_TRADE_LOW_PRICE") or row.get("CH_LOW_PRICE") or row.get("LOW"),
                "Close": row.get("CH_CLOSING_PRICE") or row.get("CH_LAST_TRADED_PRICE") or row.get("CLOSE"),
                "Volume": row.get("CH_TOT_TRADED_QTY") or row.get("CH_TOT_TRADED_QTY ") or row.get("VOLUME") or row.get("TOTTRDQTY"),
            }
        )
    return _normalize_ohlcv(pd.DataFrame(recs))


def fetch_nse_history(ticker: str, start, end) -> pd.DataFrame:
    """Official NSE cash-market daily bars, unadjusted traded prices."""
    sym = nse_symbol(ticker)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    frames = []
    cursor = start_ts
    # NSE history windows are short; walk forward in ~80-day chunks.
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=80), end_ts)
        params = {
            "symbol": sym,
            "series": '["EQ"]',
            "from": cursor.strftime("%d-%m-%Y"),
            "to": chunk_end.strftime("%d-%m-%Y"),
        }
        payload = _nse_get(NSE_HISTORY, params)
        if payload is None:
            payload = _nse_get(NSE_HISTORY_OR, {"symbol": sym, "from": params["from"], "to": params["to"]})
        rows = []
        if isinstance(payload, dict):
            rows = payload.get("data") or payload.get("dataList") or []
        elif isinstance(payload, list):
            rows = payload
        part = _history_rows_to_df(rows)
        if not part.empty:
            frames.append(part)
        cursor = chunk_end + pd.Timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[(out.index >= start_ts) & (out.index <= end_ts)]


def fetch_stooq_history(ticker: str, start, end) -> pd.DataFrame:
    """Stooq exchange EOD CSV. No key."""
    sym = stooq_symbol(ticker)
    urls = [
        f"https://stooq.com/q/d/l/?s={sym}&i=d",
        f"http://stooq.com/q/d/l/?s={sym}&i=d",
    ]
    text = ""
    for url in urls:
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            if r.status_code == 200 and r.text and "Date" in r.text[:80]:
                text = r.text
                break
        except Exception:
            continue
    if not text:
        return pd.DataFrame()
    try:
        raw = pd.read_csv(io.StringIO(text))
    except Exception:
        return pd.DataFrame()
    out = _normalize_ohlcv(raw)
    if out.empty:
        return out
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    return out[(out.index >= start_ts) & (out.index <= end_ts)]


def fetch_yahoo_history(ticker: str, start, end) -> pd.DataFrame:
    """Unofficial delayed scrape — used only if NSE/Stooq fail."""
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    try:
        df = yf.download(
            ticker,
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = [str(x) for x in df.columns.get_level_values(0)]
        if any(x.lower() in {"open", "high", "low", "close", "volume", "adj close"} for x in lvl0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    # Prefer raw Close (actual session close), not adjusted
    return _normalize_ohlcv(df)


def apply_live_quote(df: pd.DataFrame, quote: dict) -> pd.DataFrame:
    """If NSE LTP is for today, splice it onto the last bar (or append)."""
    if df is None or df.empty or not quote:
        return df
    last = quote.get("lastPrice")
    if last is None or not math.isfinite(float(last)):
        return df
    today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    row = {
        "Open": float(quote.get("open") or last),
        "High": max(float(quote.get("high") or last), float(last)),
        "Low": min(float(quote.get("low") or last), float(last)),
        "Close": float(last),
        "Volume": float(quote.get("volume") or (df["Volume"].iloc[-1] if len(df) else 0)),
    }
    out = df.copy()
    cols = ["Open", "High", "Low", "Close", "Volume"]
    values = [row[c] for c in cols]
    last_day = out.index[-1].normalize()
    if last_day == today:
        out.loc[out.index[-1], cols] = values
    elif last_day < today:
        out.loc[today, cols] = values
    return out.sort_index()


def _base_info(ticker: str) -> dict:
    meta = COMPANY_META.get(ticker) or {}
    return {
        "longName": meta.get("longName") or TICKER_NAMES.get(ticker, ticker),
        "sector": meta.get("sector") or "N/A",
        "industry": meta.get("industry") or "N/A",
    }


def fetch_market(ticker: str, start, end) -> dict:
    """
    Resolve the best free *real* source for this ticker.

    Returns dict: ohlcv, source, source_id, info, quote, note, live
    """
    ticker = (ticker or "").strip()
    info = _base_info(ticker)
    quote: dict = {}
    note = ""
    live = False
    df = pd.DataFrame()
    source = ""
    source_id = ""

    if is_india_ticker(ticker):
        quote = fetch_nse_quote(ticker)
        if quote:
            info["longName"] = quote.get("longName") or info["longName"]
            info["sector"] = quote.get("sector") or info["sector"]
            info["industry"] = quote.get("industry") or info["industry"]
        df = fetch_nse_history(ticker, start, end)
        if not df.empty:
            source = "NSE India official (unadjusted traded OHLC)"
            source_id = "nse"
            if quote:
                df = apply_live_quote(df, quote)
                live = True
                note = "Last bar uses NSE last-traded price when the session is today."
        if df.empty:
            df = fetch_stooq_history(ticker, start, end)
            if not df.empty:
                source = "Stooq EOD (India .in) — Yahoo not used"
                source_id = "stooq"
    else:
        df = fetch_stooq_history(ticker, start, end)
        if not df.empty:
            source = "Stooq EOD (exchange session close)"
            source_id = "stooq"

    if df.empty:
        df = fetch_yahoo_history(ticker, start, end)
        if not df.empty:
            source = "Yahoo Finance (unofficial, delayed) — fallback only"
            source_id = "yahoo"
            note = "Yahoo is not exchange data. Prefer NSE/Stooq; this path is last resort."

    return {
        "ohlcv": df,
        "source": source or "none",
        "source_id": source_id or "none",
        "info": info,
        "quote": quote,
        "note": note,
        "live": live,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def heatmap_returns(tickers: tuple[str, ...]) -> pd.DataFrame:
    """~5-session % change from NSE LTP pChange when possible, else history."""
    rows = []
    today = pd.Timestamp.now().normalize()
    start = today - pd.Timedelta(days=21)
    for t in tickers:
        change = None
        lookback = 5
        if is_india_ticker(t):
            q = fetch_nse_quote(t)
            if q and q.get("pChange") is not None:
                # Official day's % change is 1-session; still better than Yahoo.
                # For 5-session, prefer history.
                hist = fetch_nse_history(t, start, today)
                if len(hist) >= 2:
                    lb = min(5, len(hist) - 1)
                    base = float(hist["Close"].iloc[-1 - lb])
                    last = float(q.get("lastPrice") or hist["Close"].iloc[-1])
                    if base:
                        change = (last - base) / base * 100
                        lookback = lb
                if change is None:
                    change = float(q["pChange"])
                    lookback = 1
        if change is None:
            hist = fetch_stooq_history(t, start, today)
            if hist.empty:
                hist = fetch_yahoo_history(t, start, today)
            if len(hist) >= 2:
                lb = min(5, len(hist) - 1)
                base = float(hist["Close"].iloc[-1 - lb])
                last = float(hist["Close"].iloc[-1])
                if base:
                    change = (last - base) / base * 100
                    lookback = lb
        if change is None:
            continue
        rows.append({"Ticker": t, "Change": round(float(change), 2), "Lookback": int(lookback)})
    return pd.DataFrame(rows)
