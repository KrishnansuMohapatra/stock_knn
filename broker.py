"""
Simulated discount-broker mechanics on free daily data.

Not a live gateway. Ticket fills are last close + slippage.
The historical simulator (backtest.py) fills at next session open.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import floor

from config import (
    ATR_STOP_MULT,
    DEFAULT_CASH_INR,
    DEFAULT_CASH_USD,
    DEFAULT_RISK_PCT,
    MAX_NOTIONAL_PCT,
    TAKE_PROFIT_R,
)
from utils import currency_for, money_symbol, safe_float


@dataclass(frozen=True)
class CostModel:
    name: str
    brokerage_pct: float
    stt_sell_pct: float
    exchange_pct: float
    slippage_pct: float


COST_PRESETS = {
    "india": CostModel(
        name="India delivery (approx. discount broker)",
        brokerage_pct=0.0003,
        stt_sell_pct=0.001,
        exchange_pct=0.000035,
        slippage_pct=0.0005,
    ),
    "us": CostModel(
        name="US zero-commission (approx.)",
        brokerage_pct=0.0,
        stt_sell_pct=0.0,
        exchange_pct=0.0,
        slippage_pct=0.0005,
    ),
}


def cost_model_for(ticker: str) -> CostModel:
    return COST_PRESETS["india"] if currency_for(ticker) == "INR" else COST_PRESETS["us"]


def fee_on_fill(notional: float, side: str, model: CostModel) -> float:
    notional = abs(float(notional))
    fee = notional * (model.brokerage_pct + model.exchange_pct)
    if side.upper() in {"SELL", "SHORT"}:
        fee += notional * model.stt_sell_pct
    return fee


def slipped_price(price: float, side: str, model: CostModel) -> float:
    price = float(price)
    if side.upper() in {"BUY", "COVER"}:
        return price * (1.0 + model.slippage_pct)
    return price * (1.0 - model.slippage_pct)


def empty_book(cash_inr: float = DEFAULT_CASH_INR, cash_usd: float = DEFAULT_CASH_USD) -> dict:
    return {
        "cash": {"INR": float(cash_inr), "USD": float(cash_usd)},
        "positions": {},
        "trades": [],
    }


def equity_by_ccy(book: dict, prices: dict[str, float] | None = None) -> dict[str, float]:
    prices = prices or {}
    eq = {
        "INR": float(book["cash"].get("INR", 0.0)),
        "USD": float(book["cash"].get("USD", 0.0)),
    }
    for ticker, pos in book.get("positions", {}).items():
        ccy = currency_for(ticker)
        qty = float(pos.get("qty", 0))
        px = safe_float(prices.get(ticker), default=float(pos.get("avg", 0)))
        eq[ccy] += qty * px
    return eq


def position_size(
    price: float,
    stop: float,
    equity: float,
    cash: float,
    model: CostModel,
    risk_pct: float = DEFAULT_RISK_PCT,
    side: str = "BUY",
) -> int:
    price = safe_float(price, 0.0)
    if price <= 0 or equity <= 0 or cash <= 0:
        return 0
    stop_dist = abs(price - safe_float(stop, price))
    risk_amount = max(0.0, equity * float(risk_pct))
    qty_risk = floor(risk_amount / stop_dist) if stop_dist > 1e-9 else 0

    fill = slipped_price(price, side, model)
    rate = model.brokerage_pct + model.exchange_pct
    if side.upper() == "SELL":
        rate += model.stt_sell_pct
    denom = fill * (1.0 + rate)
    cap = cash * MAX_NOTIONAL_PCT
    qty_cash = floor(cap / denom) if denom > 0 else 0
    qty = min(qty_risk, qty_cash) if qty_risk > 0 else qty_cash
    return max(0, int(qty))


def suggest_trade(
    ticker: str,
    price: float,
    atr: float,
    score: int,
    equity: float,
    cash: float,
    held_qty: int = 0,
    risk_pct: float = DEFAULT_RISK_PCT,
    stop_mult: float = ATR_STOP_MULT,
    target_r: float = TAKE_PROFIT_R,
) -> dict:
    model = cost_model_for(ticker)
    price = safe_float(price, 0.0)
    atr = max(safe_float(atr, 0.0), 0.0)
    stop_dist = stop_mult * atr if atr > 0 else price * 0.02

    if score >= 15:
        side = "BUY"
        stop = price - stop_dist
        target = price + target_r * stop_dist
        qty = position_size(price, stop, equity, cash, model, risk_pct, "BUY")
    elif score <= -15:
        side = "SELL"
        stop = price + stop_dist
        target = price - target_r * stop_dist
        qty = max(int(held_qty), 0)
        if qty <= 0:
            qty = position_size(price, stop, equity, cash, model, risk_pct, "SELL")
    else:
        side = "HOLD"
        stop = price - stop_dist
        target = price + target_r * stop_dist
        qty = 0

    fill_side = "BUY" if side != "SELL" else "SELL"
    fill = slipped_price(price, fill_side, model) if side != "HOLD" else price
    notional = fill * qty
    fee = fee_on_fill(notional, fill_side, model) if qty else 0.0
    risk_cash = abs(price - stop) * qty if qty else 0.0

    return {
        "side": side,
        "qty": int(qty),
        "price": price,
        "fill": fill,
        "stop": stop,
        "target": target,
        "fee": fee,
        "notional": notional,
        "risk_cash": risk_cash,
        "model_name": model.name,
        "symbol": money_symbol(ticker),
        "ccy": currency_for(ticker),
        "stop_dist": stop_dist,
    }


def execute(
    book: dict,
    ticker: str,
    action: str,
    qty: int,
    raw_price: float,
    stop=None,
    target=None,
    note: str = "paper @ last close",
) -> tuple[dict, str]:
    """Long-only paper blotter. SELL reduces an existing long; no naked shorts."""
    book = deepcopy(book)
    action = (action or "").upper().strip()
    qty = int(qty)
    price = safe_float(raw_price, 0.0)
    if qty <= 0 or price <= 0:
        return book, "Quantity and price must be positive."
    if action not in {"BUY", "SELL"}:
        return book, "Paper blotter is long-only. Use BUY or SELL."

    model = cost_model_for(ticker)
    ccy = currency_for(ticker)
    pos = book["positions"].get(
        ticker, {"qty": 0.0, "avg": 0.0, "stop": None, "target": None}
    )
    current = float(pos.get("qty", 0.0))
    fill = slipped_price(price, action, model)
    notional = fill * qty
    fee = fee_on_fill(notional, action, model)

    if action == "BUY":
        total = notional + fee
        if book["cash"][ccy] < total:
            return book, f"Insufficient {ccy} cash (need {total:,.2f})."
        new_qty = current + qty
        avg = (current * float(pos.get("avg", 0.0)) + notional) / new_qty
        book["cash"][ccy] -= total
        pos = {"qty": float(new_qty), "avg": float(avg), "stop": stop, "target": target}
    else:
        if current <= 0:
            return book, "No long position to sell."
        qty = min(qty, int(current))
        notional = fill * qty
        fee = fee_on_fill(notional, "SELL", model)
        book["cash"][ccy] += notional - fee
        new_qty = current - qty
        if new_qty <= 0:
            pos = {"qty": 0.0, "avg": 0.0, "stop": None, "target": None}
        else:
            pos = {**pos, "qty": float(new_qty)}

    if abs(float(pos["qty"])) < 1e-9:
        book["positions"].pop(ticker, None)
    else:
        book["positions"][ticker] = pos

    book["trades"].append(
        {
            "ticker": ticker,
            "action": action,
            "qty": int(qty),
            "price": float(price),
            "fill": float(fill),
            "fee": float(fee),
            "note": note,
        }
    )
    return book, "ok"
