from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd

from .backtester import run_backtest
from .config import load_config
from .data import load_daily
from .features import prepare_features
from .labels import build_research_labels
from .performance import summarize_performance
from .signals import build_signals


DEFAULT_SWEEP = {
    "candidate_top_n": [30, 50],
    "trigger_amount_to_ma60": [1.3, 1.5],
    "stop_loss": [-0.08, -0.10],
    "max_holding_days": [20, 30],
}


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return default
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if not value:
        return default
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def update_cfg_for_params(base_cfg: dict, params: dict) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["signal"]["candidate_top_n"] = int(params["candidate_top_n"])
    cfg["signal"]["trigger_amount_to_ma60"] = float(params["trigger_amount_to_ma60"])
    cfg["risk"]["stop_loss"] = float(params["stop_loss"])
    cfg["risk"]["max_holding_days"] = int(params["max_holding_days"])
    return cfg


def iter_param_grid(grid: dict[str, Iterable]) -> Iterable[dict]:
    keys = list(grid.keys())
    for values in product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values, strict=True))


def run_parameter_sweep(
    data_path: str | Path,
    config_path: str | Path,
    out_dir: str | Path = "data/param_sweep",
    mode: str = "confirmed",
    start: str | None = None,
    end: str | None = None,
    include_labels: bool = True,
    grid: dict | None = None,
) -> pd.DataFrame:
    base_cfg = load_config(config_path)
    daily = load_daily(data_path, start or None, end or None)
    features = prepare_features(daily)
    if include_labels:
        features = build_research_labels(features)

    grid = grid or DEFAULT_SWEEP
    rows = []
    for i, params in enumerate(iter_param_grid(grid), start=1):
        cfg = update_cfg_for_params(base_cfg, params)
        signals = build_signals(features, cfg, mode)
        result = run_backtest(signals, cfg, mode)
        metrics = summarize_performance(result, cfg)
        rows.append(flatten_sweep_row(i, mode, params, metrics))

    result_df = pd.DataFrame(rows).sort_values(
        ["can_trade_score", "total_return", "max_drawdown"],
        ascending=[False, False, False],
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / "sweep_results.csv", index=False, encoding="utf-8-sig")
    if not result_df.empty:
        result_df.head(20).to_csv(out_dir / "sweep_top20.csv", index=False, encoding="utf-8-sig")
    return result_df


def flatten_sweep_row(index: int, mode: str, params: dict, metrics: dict) -> dict:
    row = {
        "run_id": index,
        "mode": mode,
        **params,
    }
    wanted = [
        "total_return",
        "annual_return",
        "max_drawdown",
        "sharpe",
        "calmar",
        "turnover",
        "filled_trade_count",
        "round_trip_count",
        "win_rate",
        "avg_trade_return",
        "profit_loss_ratio",
        "buy_signal_count",
        "filled_buy_count",
        "tradable_buy_rate",
        "limit_buy_block_count",
        "limit_sell_block_count",
        "top10_precision_start_10d",
        "top10_leader_hit_rate_20d",
        "top10_avg_future_max_return_20d",
        "top10_avg_lead_days",
    ]
    for key in wanted:
        row[key] = metrics.get(key)

    total_return = row.get("total_return") or 0.0
    max_drawdown = abs(row.get("max_drawdown") or 0.0)
    tradable = row.get("tradable_buy_rate") or 0.0
    trades = row.get("round_trip_count") or 0
    row["return_to_drawdown"] = total_return / max_drawdown if max_drawdown else 0.0
    row["can_trade_score"] = 1 if trades > 0 and tradable > 0 else 0
    return row
