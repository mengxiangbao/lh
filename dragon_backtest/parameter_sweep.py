from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd

from .backtester import run_backtest
from .config import load_config
from .data import compute_data_hash, load_daily
from .features import prepare_features_cached
from .labels import build_research_labels
from .performance import summarize_performance
from .signals import build_signals


DEFAULT_SWEEP = {
    "candidate_top_n": [30, 50],
    "trigger_amount_to_ma60": [1.3, 1.5],
    "stop_loss": [-0.08, -0.10],
    "max_holding_days": [20, 30],
}

CORE_METRIC_KEYS = [
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
    include_labels: bool = False,
    grid: dict | None = None,
    feature_cache_dir: str | None = "data/feature_store",
    compress_float32: bool = False,
    walk_forward: bool = False,
    train_months: int = 24,
    test_months: int = 3,
) -> pd.DataFrame:
    base_cfg = load_config(config_path)
    resolved_data_path = Path(data_path)
    daily = load_daily(resolved_data_path, start or None, end or None)
    data_hash = compute_data_hash(resolved_data_path)
    features, _ = prepare_features_cached(
        daily=daily,
        cache_dir=feature_cache_dir,
        data_hash=data_hash,
        float32=compress_float32,
    )
    if include_labels:
        features = build_research_labels(features)

    grid = grid or DEFAULT_SWEEP
    rows = []
    for i, params in enumerate(iter_param_grid(grid), start=1):
        cfg = update_cfg_for_params(base_cfg, params)
        if walk_forward:
            row = run_walk_forward_for_params(
                run_id=i,
                mode=mode,
                params=params,
                cfg=cfg,
                features=features,
                train_months=train_months,
                test_months=test_months,
            )
        else:
            signals = build_signals(features, cfg, mode)
            result = run_backtest(signals, cfg, mode)
            metrics = summarize_performance(result, cfg)
            row = flatten_sweep_row(i, mode, params, metrics)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    sort_cols = ["can_trade_score", "total_return", "max_drawdown"]
    sort_asc = [False, False, False]
    if walk_forward:
        sort_cols = ["wf_can_trade_score", "wf_out_total_return", "wf_out_max_drawdown_abs"]
        sort_asc = [False, False, True]
    result_df = result_df.sort_values(sort_cols, ascending=sort_asc)
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
    for key in CORE_METRIC_KEYS:
        row[key] = metrics.get(key)

    total_return = row.get("total_return") or 0.0
    max_drawdown = abs(row.get("max_drawdown") or 0.0)
    tradable = row.get("tradable_buy_rate") or 0.0
    trades = row.get("round_trip_count") or 0
    row["return_to_drawdown"] = total_return / max_drawdown if max_drawdown else 0.0
    row["can_trade_score"] = 1 if trades > 0 and tradable > 0 else 0
    return row


def run_walk_forward_for_params(
    run_id: int,
    mode: str,
    params: dict,
    cfg: dict,
    features: pd.DataFrame,
    train_months: int,
    test_months: int,
) -> dict:
    out = {"run_id": run_id, "mode": mode, **params}
    windows = build_walk_forward_windows(features, train_months=train_months, test_months=test_months)
    if not windows:
        out["wf_window_count"] = 0
        out["wf_can_trade_score"] = 0
        return out

    window_rows = []
    in_total_returns = []
    out_total_returns = []
    out_drawdowns_abs = []
    out_sharpes = []
    for win_idx, (train_start, train_end, test_start, test_end) in enumerate(windows, start=1):
        train = features[(features["date"] >= train_start) & (features["date"] <= train_end)].copy()
        test = features[(features["date"] >= test_start) & (features["date"] <= test_end)].copy()
        if train.empty or test.empty:
            continue

        in_metrics = summarize_performance(run_backtest(build_signals(train, cfg, mode), cfg, mode), cfg)
        out_metrics = summarize_performance(run_backtest(build_signals(test, cfg, mode), cfg, mode), cfg)
        in_total_returns.append(float(in_metrics.get("total_return", 0.0) or 0.0))
        out_total_returns.append(float(out_metrics.get("total_return", 0.0) or 0.0))
        out_dd = float(out_metrics.get("max_drawdown", 0.0) or 0.0)
        out_drawdowns_abs.append(abs(out_dd))
        out_sharpes.append(float(out_metrics.get("sharpe", 0.0) or 0.0))
        window_rows.append(
            {
                "window_id": win_idx,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "in_total_return": in_total_returns[-1],
                "out_total_return": out_total_returns[-1],
                "out_max_drawdown": out_dd,
                "out_sharpe": out_sharpes[-1],
            }
        )

    if not window_rows:
        out["wf_window_count"] = 0
        out["wf_can_trade_score"] = 0
        return out

    out["wf_window_count"] = len(window_rows)
    out["wf_in_total_return"] = _mean(in_total_returns)
    out["wf_out_total_return"] = _mean(out_total_returns)
    out["wf_out_max_drawdown_abs"] = _mean(out_drawdowns_abs)
    out["wf_out_sharpe"] = _mean(out_sharpes)
    out["wf_stability_std_out_return"] = _std(out_total_returns)
    out["wf_return_decay"] = out["wf_out_total_return"] - out["wf_in_total_return"]
    out["wf_positive_out_rate"] = float(sum(x > 0 for x in out_total_returns) / len(out_total_returns))
    out["wf_can_trade_score"] = 1 if out["wf_window_count"] > 0 else 0
    return out


def build_walk_forward_windows(
    features: pd.DataFrame,
    train_months: int = 24,
    test_months: int = 3,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if features.empty:
        return []
    dates = pd.to_datetime(features["date"]).dropna().sort_values()
    if dates.empty:
        return []
    first = dates.min().to_period("M")
    last = dates.max().to_period("M")
    month_starts = pd.period_range(first, last, freq="M")
    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    i = train_months
    while i + test_months <= len(month_starts):
        train_start_period = month_starts[i - train_months]
        train_end_period = month_starts[i - 1]
        test_start_period = month_starts[i]
        test_end_period = month_starts[i + test_months - 1]
        train_start = train_start_period.to_timestamp(how="start")
        train_end = train_end_period.to_timestamp(how="end")
        test_start = test_start_period.to_timestamp(how="start")
        test_end = test_end_period.to_timestamp(how="end")
        windows.append((train_start, train_end, test_start, test_end))
        i += test_months
    return windows


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    s = pd.Series(values, dtype=float)
    return float(s.std(ddof=0))
