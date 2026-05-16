from __future__ import annotations

from dataclasses import dataclass
from math import floor, sqrt

import pandas as pd


@dataclass
class Position:
    code: str
    shares: int
    entry_date: pd.Timestamp
    entry_index: int
    entry_price: float
    entry_value: float
    entry_cost: float
    highest_close: float
    last_close: float
    triggered: bool = False


def stamp_tax_rate(trade_date: pd.Timestamp, trade_cfg: dict) -> float:
    if pd.Timestamp(trade_date) >= pd.Timestamp("2023-08-28"):
        return trade_cfg["stamp_tax_rate_after_20230828"]
    return trade_cfg["stamp_tax_rate_before_20230828"]


def trade_cost(side: str, value: float, trade_date: pd.Timestamp, trade_cfg: dict) -> float:
    if value <= 0:
        return 0.0
    commission = max(value * trade_cfg["commission_rate"], trade_cfg["min_commission"])
    transfer_fee = value * trade_cfg["transfer_fee_rate"]
    stamp = value * stamp_tax_rate(trade_date, trade_cfg) if side == "sell" else 0.0
    return commission + transfer_fee + stamp


def is_one_price_limit(row: pd.Series, side: str, tol: float) -> bool:
    if side == "buy":
        limit_price = row["up_limit"]
    else:
        limit_price = row["down_limit"]
    if not has_valid_price_limit(row, side):
        return False
    prices = [row["open"], row["high"], row["low"], row["close"]]
    return all(abs(float(p) / float(limit_price) - 1) <= tol for p in prices)


def has_valid_price_limit(row: pd.Series, side: str) -> bool:
    up_limit = float(row.get("up_limit", 0) or 0)
    down_limit = float(row.get("down_limit", 0) or 0)
    pre_close = float(row.get("pre_close", 0) or 0)
    if up_limit >= 9999 or down_limit <= 0.01:
        return False
    if pre_close > 0 and up_limit / pre_close - 1 > 5:
        return False
    if side == "buy":
        return up_limit > 0
    return down_limit > 0


def can_buy(row: pd.Series, trade_cfg: dict) -> tuple[bool, str]:
    tol = trade_cfg["limit_tolerance"]
    if bool(row.get("paused", False)):
        return False, "suspended"
    if row.get("volume", 0) <= 0 or row.get("amount", 0) <= 0:
        return False, "no_volume"
    if has_valid_price_limit(row, "buy") and row["open"] >= row["up_limit"] * (1 - tol):
        return False, "limit_up_open"
    if is_one_price_limit(row, "buy", tol):
        return False, "one_price_limit_up"
    return True, ""


def can_sell(row: pd.Series, position: Position, trade_date: pd.Timestamp, trade_cfg: dict) -> tuple[bool, str]:
    tol = trade_cfg["limit_tolerance"]
    if pd.Timestamp(trade_date) <= pd.Timestamp(position.entry_date):
        return False, "t_plus_1"
    if bool(row.get("paused", False)):
        return False, "suspended"
    if row.get("volume", 0) <= 0 or row.get("amount", 0) <= 0:
        return False, "no_volume"
    if has_valid_price_limit(row, "sell") and row["open"] <= row["down_limit"] * (1 + tol):
        return False, "limit_down_open"
    if is_one_price_limit(row, "sell", tol):
        return False, "one_price_limit_down"
    return True, ""


def impact(row: pd.Series, order_value: float, trade_cfg: dict) -> float:
    amount = max(float(row.get("amount", 0.0)), 1.0)
    participation = max(order_value, 0.0) / amount
    return trade_cfg["impact_k"] * sqrt(max(participation, 0.0))


def execute_buy(
    row: pd.Series,
    signal_row: pd.Series,
    trade_date: pd.Timestamp,
    trade_index: int,
    target_value: float,
    cash: float,
    trade_cfg: dict,
) -> tuple[Position | None, float, dict]:
    ok, reason = can_buy(row, trade_cfg)
    record = {
        "trade_date": trade_date,
        "signal_date": signal_row["date"],
        "code": row["code"],
        "side": "buy",
        "status": "blocked" if not ok else "filled",
        "blocked_reason": reason,
        "reason": "entry",
        "shares": 0,
        "price": 0.0,
        "value": 0.0,
        "cost": 0.0,
        "cash_after": cash,
        "score": float(signal_row.get("score", 0.0)),
        "trigger": bool(signal_row.get("trigger", False)),
        "target_weight_mult": float(signal_row.get("target_weight_mult", 1.0)),
        "market_size_mult": float(signal_row.get("market_size_mult", 1.0)),
        "ts_code": row.get("ts_code", row.get("code", "")),
        "name": row.get("name", ""),
        "exchange": row.get("exchange", ""),
        "market": row.get("market", ""),
        "sector": row.get("sector", ""),
    }
    if not ok:
        return None, cash, record

    signal_amount = max(float(signal_row.get("amount", 0.0) or 0.0), 0.0)
    trade_amount = max(float(row.get("amount", 0.0) or 0.0), 0.0)
    capacity_base_mode = str(trade_cfg.get("capacity_base", "min_signal_trade"))
    if capacity_base_mode == "signal":
        capacity_base = signal_amount
        zero_reason = "no_signal_liquidity"
    elif capacity_base_mode == "trade":
        capacity_base = trade_amount
        zero_reason = "no_trade_liquidity"
    elif capacity_base_mode == "min_signal_trade":
        capacity_base = min(signal_amount, trade_amount)
        zero_reason = "no_capacity_liquidity"
    else:
        raise ValueError(f"Unknown trade.capacity_base: {capacity_base_mode}")

    max_value = capacity_base * trade_cfg["volume_cap"]
    order_value = min(float(target_value), max_value, cash)
    if order_value <= 0:
        reason = zero_reason if capacity_base <= 0 else "no_cash"
        record.update(status="blocked", blocked_reason=reason)
        return None, cash, record

    fill_price = float(row["open"]) * (1 + trade_cfg["buy_slippage"] + impact(row, order_value, trade_cfg))
    lot_size = int(trade_cfg["lot_size"])
    shares = floor(order_value / fill_price / lot_size) * lot_size
    while shares > 0:
        value = shares * fill_price
        cost = trade_cost("buy", value, trade_date, trade_cfg)
        if value + cost <= cash:
            break
        shares -= lot_size
    if shares <= 0:
        record.update(status="blocked", blocked_reason="below_lot_or_cash")
        return None, cash, record

    value = shares * fill_price
    cost = trade_cost("buy", value, trade_date, trade_cfg)
    cash_after = cash - value - cost
    position = Position(
        code=str(row["code"]),
        shares=int(shares),
        entry_date=pd.Timestamp(trade_date),
        entry_index=int(trade_index),
        entry_price=float(fill_price),
        entry_value=float(value),
        entry_cost=float(cost),
        highest_close=float(row["close"]),
        last_close=float(row["close"]),
        triggered=bool(signal_row.get("trigger", False)),
    )
    record.update(
        shares=int(shares),
        price=float(fill_price),
        value=float(value),
        cost=float(cost),
        cash_after=float(cash_after),
        capacity_base_mode=capacity_base_mode,
        signal_amount=float(signal_amount),
        trade_amount=float(trade_amount),
        capacity_base=float(capacity_base),
    )
    return position, cash_after, record


def execute_sell(
    row: pd.Series,
    position: Position,
    trade_date: pd.Timestamp,
    cash: float,
    trade_cfg: dict,
    reason: str,
) -> tuple[bool, float, dict]:
    ok, blocked_reason = can_sell(row, position, trade_date, trade_cfg)
    record = {
        "trade_date": trade_date,
        "signal_date": pd.NaT,
        "code": position.code,
        "side": "sell",
        "status": "blocked" if not ok else "filled",
        "blocked_reason": blocked_reason,
        "reason": reason,
        "shares": 0,
        "price": 0.0,
        "value": 0.0,
        "cost": 0.0,
        "cash_after": cash,
        "score": 0.0,
        "trigger": position.triggered,
        "ts_code": row.get("ts_code", row.get("code", "")),
        "name": row.get("name", ""),
        "exchange": row.get("exchange", ""),
        "market": row.get("market", ""),
        "pnl": 0.0,
        "return": 0.0,
        "holding_days": None,
        "sector": row.get("sector", ""),
    }
    if not ok:
        return False, cash, record

    rough_value = position.shares * float(row["open"])
    fill_price = float(row["open"]) * (1 - trade_cfg["sell_slippage"] - impact(row, rough_value, trade_cfg))
    value = position.shares * fill_price
    cost = trade_cost("sell", value, trade_date, trade_cfg)
    cash_after = cash + value - cost
    pnl = value - cost - position.entry_value - position.entry_cost
    ret = pnl / max(position.entry_value + position.entry_cost, 1.0)
    record.update(
        shares=int(position.shares),
        price=float(fill_price),
        value=float(value),
        cost=float(cost),
        cash_after=float(cash_after),
        pnl=float(pnl),
    )
    record["return"] = float(ret)
    return True, cash_after, record
