"""
End-of-day signal backtest with next-open fills, costs, slippage, ATR stops.

Decision at close[t] → fill at open[t+1] (the first price a real broker could
give you after an EOD signal). No paid data required.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from broker import CostModel, cost_model_for, fee_on_fill, slipped_price
from config import ATR_STOP_MULT, DEFAULT_RISK_PCT, MAX_NOTIONAL_PCT, TAKE_PROFIT_R
from signals import compute_signal_score, first_signal_index


def _cagr(total_return: float, n_days: int) -> float:
    if n_days <= 0 or total_return <= -1:
        return float("nan")
    years = n_days / 252.0
    if years <= 0:
        return float("nan")
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak.replace(0, np.nan) - 1.0
    return float(dd.min())


def _sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 5 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * math.sqrt(252))


def run_signal_backtest(
    data: pd.DataFrame,
    rsi_ob: int,
    rsi_os: int,
    ticker: str,
    initial_capital: float,
    long_short: bool = False,
    risk_pct: float = DEFAULT_RISK_PCT,
    buy_threshold: int = 15,
    sell_threshold: int = -15,
    atr_stop_mult: float = ATR_STOP_MULT,
    target_r: float = TAKE_PROFIT_R,
) -> dict:
    model: CostModel = cost_model_for(ticker)
    start = first_signal_index(data)
    if start is None or start >= len(data) - 2:
        return {"error": "Not enough bars with valid indicators to backtest."}
    if initial_capital <= 0:
        return {"error": "Initial capital must be positive."}

    cash = float(initial_capital)
    qty = 0  # signed: + long, − short
    avg = 0.0
    stop = None
    target = None
    pending = None  # ("BUY"|"SELL"|"FLAT", decision_index)

    equity_rows = []
    trade_rows = []

    def mark(i: int) -> float:
        close = float(data["Close"].iloc[i])
        return cash + qty * close

    def open_fill(i_open: int, side: str, shares: int) -> tuple[float, float]:
        raw = float(data["Open"].iloc[i_open])
        fill = slipped_price(raw, side, model)
        fee = fee_on_fill(fill * shares, side, model)
        return fill, fee

    n = len(data)
    for i in range(start, n):
        o = float(data["Open"].iloc[i])
        h = float(data["High"].iloc[i])
        l = float(data["Low"].iloc[i])
        c = float(data["Close"].iloc[i])
        atr = float(data["ATR"].iloc[i]) if pd.notna(data["ATR"].iloc[i]) else 0.0

        # 1) Fill orders queued at previous close
        if pending is not None:
            action, _dec_i, want_signed = pending
            pending = None
            if action == "FLAT" and qty != 0:
                side = "SELL" if qty > 0 else "BUY"
                shares = abs(qty)
                fill, fee = open_fill(i, side, shares)
                if qty > 0:
                    cash += fill * shares - fee
                else:
                    cash -= fill * shares + fee
                trade_rows.append(
                    {
                        "date": data.index[i],
                        "action": "EXIT",
                        "qty": shares,
                        "fill": fill,
                        "fee": fee,
                    }
                )
                qty, avg, stop, target = 0, 0.0, None, None
            elif action in {"BUY", "SHORT"} and qty == 0 and want_signed != 0:
                side = "BUY" if want_signed > 0 else "SHORT"
                shares = abs(int(want_signed))
                fill, fee = open_fill(i, "BUY" if side == "BUY" else "SELL", shares)
                cost = fill * shares + fee
                if side == "BUY":
                    if cash >= cost:
                        cash -= cost
                        qty = shares
                        avg = fill
                        stop = fill - atr_stop_mult * atr if atr > 0 else fill * 0.98
                        target = fill + target_r * (fill - stop)
                        trade_rows.append(
                            {
                                "date": data.index[i],
                                "action": "BUY",
                                "qty": shares,
                                "fill": fill,
                                "fee": fee,
                            }
                        )
                else:
                    # short: credit proceeds, pay fee
                    cash += fill * shares - fee
                    qty = -shares
                    avg = fill
                    stop = fill + atr_stop_mult * atr if atr > 0 else fill * 1.02
                    target = fill - target_r * (stop - fill)
                    trade_rows.append(
                        {
                            "date": data.index[i],
                            "action": "SHORT",
                            "qty": shares,
                            "fill": fill,
                            "fee": fee,
                        }
                    )

        # 2) Intrabar stop / target using the day's range (conservative: stop first)
        if qty > 0 and stop is not None:
            hit_stop = l <= stop
            hit_tgt = h >= (target if target is not None else math.inf)
            if hit_stop:
                fill = slipped_price(min(o, stop), "SELL", model)
                fee = fee_on_fill(fill * qty, "SELL", model)
                cash += fill * qty - fee
                trade_rows.append(
                    {"date": data.index[i], "action": "STOP", "qty": qty, "fill": fill, "fee": fee}
                )
                qty, avg, stop, target = 0, 0.0, None, None
                pending = None
            elif hit_tgt and target is not None:
                fill = slipped_price(max(o, target), "SELL", model)
                fee = fee_on_fill(fill * qty, "SELL", model)
                cash += fill * qty - fee
                trade_rows.append(
                    {"date": data.index[i], "action": "TARGET", "qty": qty, "fill": fill, "fee": fee}
                )
                qty, avg, stop, target = 0, 0.0, None, None
                pending = None
        elif qty < 0 and stop is not None:
            shares = abs(qty)
            hit_stop = h >= stop
            hit_tgt = l <= (target if target is not None else -math.inf)
            if hit_stop:
                fill = slipped_price(max(o, stop), "BUY", model)
                fee = fee_on_fill(fill * shares, "BUY", model)
                cash -= fill * shares + fee
                trade_rows.append(
                    {"date": data.index[i], "action": "STOP", "qty": shares, "fill": fill, "fee": fee}
                )
                qty, avg, stop, target = 0, 0.0, None, None
                pending = None
            elif hit_tgt and target is not None:
                fill = slipped_price(min(o, target), "BUY", model)
                fee = fee_on_fill(fill * shares, "BUY", model)
                cash -= fill * shares + fee
                trade_rows.append(
                    {"date": data.index[i], "action": "TARGET", "qty": shares, "fill": fill, "fee": fee}
                )
                qty, avg, stop, target = 0, 0.0, None, None
                pending = None

        # 3) Mark to close
        equity_rows.append((data.index[i], mark(i)))

        # 4) New decision at close — queued for next open (no look-ahead)
        if i >= n - 1:
            continue
        _, score = compute_signal_score(data, rsi_ob, rsi_os, i)
        if score >= buy_threshold:
            desired = 1
        elif score <= sell_threshold:
            desired = -1 if long_short else 0
        else:
            desired = 1 if qty > 0 else (-1 if qty < 0 else 0)

        if desired == 0 and qty != 0:
            pending = ("FLAT", i, 0)
        elif desired != 0 and qty == 0:
            eq = mark(i)
            atr_now = atr if atr > 0 else c * 0.02
            stop_dist = atr_stop_mult * atr_now
            risk_amount = eq * risk_pct
            shares_risk = int(risk_amount / stop_dist) if stop_dist > 1e-9 else 0
            next_open = float(data["Open"].iloc[i + 1])
            cap = eq * MAX_NOTIONAL_PCT
            shares_cash = int(cap / (next_open * 1.002)) if next_open > 0 else 0
            shares = max(0, min(shares_risk, shares_cash) if shares_risk else shares_cash)
            if shares > 0:
                pending = ("BUY" if desired > 0 else "SHORT", i, desired * shares)
        elif desired > 0 and qty < 0:
            pending = ("FLAT", i, 0)
        elif desired < 0 and qty > 0:
            pending = ("FLAT", i, 0)

    equity = pd.Series(
        {d: v for d, v in equity_rows},
        dtype=float,
        name="strategy",
    )
    if equity.empty:
        return {"error": "Backtest produced no equity points."}

    # Buy-and-hold from first decision bar's next open
    bh_start = start + 1 if start + 1 < n else start
    bh_px0 = float(data["Open"].iloc[bh_start])
    bh_shares = int((initial_capital * MAX_NOTIONAL_PCT) / bh_px0) if bh_px0 > 0 else 0
    bh_cash0 = initial_capital - bh_shares * bh_px0
    bh = pd.Series(
        {data.index[i]: bh_cash0 + bh_shares * float(data["Close"].iloc[i]) for i in range(bh_start, n)},
        dtype=float,
        name="buy_hold",
    )
    equity = equity.reindex(bh.index).ffill()

    strat_ret = float(equity.iloc[-1] / initial_capital - 1.0)
    bh_ret = float(bh.iloc[-1] / initial_capital - 1.0) if len(bh) else float("nan")
    n_days = max(len(equity) - 1, 1)
    daily = equity.pct_change()

    trades = pd.DataFrame(trade_rows)
    wins = 0
    losses = 0
    if not trades.empty:
        # pair BUY/SHORT with next EXIT/STOP/TARGET
        stack = None
        for _, row in trades.iterrows():
            if row["action"] in {"BUY", "SHORT"}:
                stack = row
            elif stack is not None and row["action"] in {"EXIT", "STOP", "TARGET"}:
                if stack["action"] == "BUY":
                    pnl = (row["fill"] - stack["fill"]) * row["qty"] - stack["fee"] - row["fee"]
                else:
                    pnl = (stack["fill"] - row["fill"]) * row["qty"] - stack["fee"] - row["fee"]
                if pnl >= 0:
                    wins += 1
                else:
                    losses += 1
                stack = None
    closed = wins + losses
    win_rate = (wins / closed * 100) if closed else float("nan")

    stats = {
        "initial": float(initial_capital),
        "final": float(equity.iloc[-1]),
        "return_pct": strat_ret * 100,
        "buy_hold_pct": bh_ret * 100,
        "cagr_pct": _cagr(strat_ret, n_days) * 100,
        "max_dd_pct": _max_drawdown(equity) * 100,
        "sharpe": _sharpe(daily),
        "trades": int(len(trades)),
        "round_trips": int(closed),
        "win_rate_pct": win_rate,
        "bars": int(n_days),
        "cost_name": model.name,
    }
    return {
        "equity": equity,
        "buy_hold": bh,
        "trades": trades,
        "stats": stats,
    }
