from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tomllib
from typing import Any


DEFAULT_CONFIG = {
    "data": {
        "daily_path": "data/raw/daily_price.csv",
    },
    "universe": {
        "min_list_days": 120,
        "exclude_st": True,
        "exclude_suspended": True,
        "min_avg_amount_20d": 100_000_000.0,
        "max_ret_20d": 0.30,
        "max_limit_up_count_10d": 1,
        "max_amount_ma5_to_ma60": 3.0,
        "min_close_to_high_60d": 0.90,
        "min_sector_shift_pct": 0.70,
        "require_stock_stronger_than_sector": True,
        "max_drawdown_vs_sector_median": True,
        "min_up_amount_ratio_20d": 1.20,
    },
    "score": {
        "weights": {
            "hidden_rs": 0.25,
            "accumulation": 0.20,
            "vol_squeeze": 0.15,
            "anti_fall": 0.15,
            "sector_shift": 0.15,
            "position": 0.10,
        }
    },
    "signal": {
        "mode": "confirmed",
        "candidate_top_n": 50,
        "candidate_top_pct": 0.01,
        "direct_top_n": 10,
        "trigger_amount_to_ma60": 1.5,
        "trigger_ret_1d": 0.05,
        "trigger_sector_ret_pct": 0.80,
        "trigger_close_location": 0.70,
        "report_top_k": 10,
    },
    "market_filter": {
        "enabled": False,
        "avoid_weak_breadth": True,
        "avoid_range_neutral": True,
        "min_positive_ratio_ma5": 0.45,
        "apply_to": "buy",
        "exit_on_fail": False,
    },
    "market_sizing": {
        "enabled": False,
        "strong_breadth_mult": 1.0,
        "neutral_breadth_mult": 0.80,
        "weak_breadth_mult": 0.50,
        "uptrend_mult": 1.0,
        "range_mult": 0.80,
        "downtrend_mult": 0.60,
        "expanding_mult": 1.0,
        "neutral_liquidity_mult": 0.90,
        "shrinking_mult": 0.80,
        "range_neutral_mult": 0.60,
        "min_mult": 0.40,
        "max_mult": 1.0,
    },
    "trade": {
        "initial_cash": 1_000_000.0,
        "target_weight": 0.10,
        "max_positions": 10,
        "capacity_base": "min_signal_trade",
        "lot_size": 100,
        "buy_slippage": 0.001,
        "sell_slippage": 0.001,
        "commission_rate": 0.00025,
        "min_commission": 5.0,
        "transfer_fee_rate": 0.00001,
        "stamp_tax_rate_before_20230828": 0.001,
        "stamp_tax_rate_after_20230828": 0.0005,
        "volume_cap": 0.05,
        "impact_k": 0.001,
        "limit_tolerance": 0.0001,
    },
    "risk": {
        "stop_loss": -0.10,
        "trailing_stop": -0.12,
        "max_holding_days": 30,
        "stale_exit_days": 10,
        "exit_ma": "ma20",
        "sector_weak_pct": 0.50,
    },
}


def deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("rb") as f:
            deep_update(cfg, tomllib.load(f))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    trade = cfg.get("trade", {})
    signal = cfg.get("signal", {})
    risk = cfg.get("risk", {})
    market_sizing = cfg.get("market_sizing", {})
    market_filter = cfg.get("market_filter", {})

    _require_between(trade, "target_weight", 0.0, 1.0, right_inclusive=True, section="trade", left_inclusive=False)
    _require_between(trade, "volume_cap", 0.0, 1.0, right_inclusive=True, section="trade", left_inclusive=False)
    _require_positive_int(trade, "max_positions", section="trade")
    _require_positive_int(trade, "lot_size", section="trade")
    _require_non_negative(trade, "buy_slippage", section="trade")
    _require_non_negative(trade, "sell_slippage", section="trade")
    _require_non_negative(trade, "commission_rate", section="trade")
    _require_non_negative(trade, "min_commission", section="trade")
    _require_non_negative(trade, "transfer_fee_rate", section="trade")
    _require_non_negative(trade, "impact_k", section="trade")
    _require_non_negative(trade, "limit_tolerance", section="trade")
    _require_enum(trade, "capacity_base", {"signal", "trade", "min_signal_trade"}, section="trade")

    _require_between(signal, "candidate_top_pct", 0.0, 1.0, section="signal", left_inclusive=True, right_inclusive=True)
    _require_positive_int(signal, "candidate_top_n", section="signal")
    _require_positive_int(signal, "direct_top_n", section="signal")
    _require_positive_int(signal, "report_top_k", section="signal")
    _require_enum(signal, "mode", {"potential", "confirmed", "hybrid"}, section="signal")

    _require_negative(risk, "stop_loss", section="risk")
    _require_negative(risk, "trailing_stop", section="risk")
    _require_positive_int(risk, "max_holding_days", section="risk")
    _require_positive_int(risk, "stale_exit_days", section="risk")
    _require_between(risk, "sector_weak_pct", 0.0, 1.0, section="risk", left_inclusive=False, right_inclusive=True)

    if market_sizing.get("enabled", False):
        _require_between(market_sizing, "min_mult", 0.0, 1.0, section="market_sizing", left_inclusive=False, right_inclusive=True)
        _require_between(market_sizing, "max_mult", 0.0, 1.0, section="market_sizing", left_inclusive=False, right_inclusive=True)
        if float(market_sizing["min_mult"]) > float(market_sizing["max_mult"]):
            raise ValueError(
                f"Invalid config: market_sizing.min_mult ({market_sizing['min_mult']}) cannot be greater than "
                f"market_sizing.max_mult ({market_sizing['max_mult']})."
            )

    if market_filter.get("enabled", False):
        _require_enum(market_filter, "apply_to", {"buy"}, section="market_filter")
        _require_between(market_filter, "min_positive_ratio_ma5", 0.0, 1.0, section="market_filter", left_inclusive=True, right_inclusive=True)


def _require_enum(d: dict[str, Any], key: str, allowed: set[str], section: str) -> None:
    value = d.get(key)
    if value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid config: {section}.{key}={value!r} is not allowed. Expected one of: {allowed_text}.")


def _require_positive_int(d: dict[str, Any], key: str, section: str) -> None:
    value = d.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid config: {section}.{key} must be a positive integer, got {value!r}.")


def _require_non_negative(d: dict[str, Any], key: str, section: str) -> None:
    value = float(d.get(key))
    if value < 0:
        raise ValueError(f"Invalid config: {section}.{key} must be >= 0, got {value!r}.")


def _require_negative(d: dict[str, Any], key: str, section: str) -> None:
    value = float(d.get(key))
    if value >= 0:
        raise ValueError(f"Invalid config: {section}.{key} must be < 0, got {value!r}.")


def _require_between(
    d: dict[str, Any],
    key: str,
    low: float,
    high: float,
    section: str,
    left_inclusive: bool = True,
    right_inclusive: bool = True,
) -> None:
    value = float(d.get(key))
    low_ok = value >= low if left_inclusive else value > low
    high_ok = value <= high if right_inclusive else value < high
    if not (low_ok and high_ok):
        left = "[" if left_inclusive else "("
        right = "]" if right_inclusive else ")"
        raise ValueError(f"Invalid config: {section}.{key}={value!r} is out of range {left}{low}, {high}{right}.")
