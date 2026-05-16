from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> tuple[float, str | None, str | None]:
    if equity.empty:
        return 0.0, None, None
    running_max = equity.cummax()
    dd = equity / running_max - 1
    end = dd.idxmin()
    start = equity.loc[:end].idxmax()
    return float(dd.loc[end]), str(start), str(end)


def summarize_performance(result: dict, cfg: dict, benchmark: pd.DataFrame | None = None) -> dict:
    equity = result["equity"].copy()
    trades = result["trades"].copy()
    signals = result["signals"].copy()

    if equity.empty:
        return {"error": "empty equity curve"}

    equity["date"] = pd.to_datetime(equity["date"])
    initial_cash = float(cfg["trade"]["initial_cash"])
    final_equity = float(equity["total_equity"].iloc[-1])
    total_return = final_equity / initial_cash - 1
    days = max(len(equity), 1)
    annual_return = (1 + total_return) ** (252 / days) - 1
    daily_ret = equity["daily_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_ret.std()) if daily_ret.std() > 0 else 0.0
    mdd, mdd_start, mdd_end = max_drawdown(equity.set_index("date")["total_equity"])
    calmar = float(annual_return / abs(mdd)) if mdd < 0 else 0.0

    filled = trades[trades.get("status", "") == "filled"] if not trades.empty else trades
    sells = filled[filled.get("side", "") == "sell"] if not filled.empty else filled
    turnover = float(filled.get("value", pd.Series(dtype=float)).sum() / equity["total_equity"].mean()) if not filled.empty else 0.0
    win_rate = float((sells.get("pnl", pd.Series(dtype=float)) > 0).mean()) if not sells.empty else 0.0
    avg_trade_return = float(sells.get("return", pd.Series(dtype=float)).mean()) if not sells.empty else 0.0
    profit_loss_ratio = _profit_loss_ratio(sells)

    blocked = trades[trades.get("status", "") == "blocked"] if not trades.empty else trades
    blocked_buy = blocked[blocked.get("side", "") == "buy"] if not blocked.empty else blocked
    blocked_sell = blocked[blocked.get("side", "") == "sell"] if not blocked.empty else blocked
    buy_signals = int(signals.get("buy_signal", pd.Series(dtype=bool)).sum())
    filled_buys = int((filled.get("side", pd.Series(dtype=str)) == "buy").sum()) if not filled.empty else 0

    topk_metrics = summarize_topk(signals, cfg["signal"]["report_top_k"])

    metrics = {
        "initial_cash": initial_cash,
        "final_equity": final_equity,
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": float(mdd),
        "max_drawdown_start": mdd_start,
        "max_drawdown_end": mdd_end,
        "sharpe": sharpe,
        "calmar": calmar,
        "turnover": turnover,
        "filled_trade_count": int(len(filled)),
        "round_trip_count": int(len(sells)),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "profit_loss_ratio": profit_loss_ratio,
        "buy_signal_count": buy_signals,
        "filled_buy_count": filled_buys,
        "tradable_buy_rate": float(filled_buys / buy_signals) if buy_signals else 0.0,
        "blocked_buy_count": int(len(blocked_buy)),
        "blocked_sell_count": int(len(blocked_sell)),
        "limit_buy_block_count": int(blocked_buy.get("blocked_reason", pd.Series(dtype=str)).str.contains("limit", na=False).sum())
        if not blocked_buy.empty
        else 0,
        "capacity_blocked_buy_count": int(
            blocked_buy.get("blocked_reason", pd.Series(dtype=str)).isin(
                ["no_signal_liquidity", "no_trade_liquidity", "no_capacity_liquidity"]
            ).sum()
        )
        if not blocked_buy.empty
        else 0,
        "minute_missing_buy_block_count": int(
            blocked_buy.get("blocked_reason", pd.Series(dtype=str)).eq("missing_minute_data").sum()
        )
        if not blocked_buy.empty
        else 0,
        "minute_no_liquidity_buy_block_count": int(
            blocked_buy.get("blocked_reason", pd.Series(dtype=str)).eq("minute_no_liquidity").sum()
        )
        if not blocked_buy.empty
        else 0,
        "minute_not_confirmed_buy_block_count": int(
            blocked_buy.get("blocked_reason", pd.Series(dtype=str)).eq("minute_not_confirmed").sum()
        )
        if not blocked_buy.empty
        else 0,
        "limit_sell_block_count": int(blocked_sell.get("blocked_reason", pd.Series(dtype=str)).str.contains("limit", na=False).sum())
        if not blocked_sell.empty
        else 0,
    }
    metrics.update(topk_metrics)
    if benchmark is not None and not benchmark.empty:
        metrics.update(summarize_benchmark(equity, benchmark))
    return metrics


def read_benchmark(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Benchmark file must include 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "benchmark_return" in df.columns:
        ret = pd.to_numeric(df["benchmark_return"], errors="coerce")
    elif "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce")
        ret = close.pct_change()
    else:
        raise ValueError("Benchmark file must include 'benchmark_return' or 'close' column.")
    out = pd.DataFrame({"date": df["date"], "benchmark_return": ret}).dropna(subset=["date"]).sort_values("date")
    return out


def summarize_benchmark(equity: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    left = equity[["date", "daily_return"]].copy()
    left["date"] = pd.to_datetime(left["date"])
    left["daily_return"] = pd.to_numeric(left["daily_return"], errors="coerce").fillna(0.0)
    right = benchmark[["date", "benchmark_return"]].copy()
    right["date"] = pd.to_datetime(right["date"])
    right["benchmark_return"] = pd.to_numeric(right["benchmark_return"], errors="coerce").fillna(0.0)

    merged = left.merge(right, on="date", how="inner").dropna()
    if merged.empty:
        return {"benchmark_overlap_days": 0}

    strat_ret = merged["daily_return"].astype(float)
    bench_ret = merged["benchmark_return"].astype(float)
    excess = strat_ret - bench_ret
    bench_total = float((1 + bench_ret).prod() - 1)
    excess_total = float((1 + strat_ret).prod() / max((1 + bench_ret).prod(), 1e-12) - 1)
    info_ratio = 0.0
    te = float(excess.std())
    if te > 0:
        info_ratio = float(np.sqrt(252) * excess.mean() / te)

    beta = 0.0
    alpha_annual = 0.0
    bench_var = float(bench_ret.var())
    if bench_var > 0:
        cov = float(np.cov(strat_ret, bench_ret, ddof=0)[0, 1])
        beta = cov / bench_var
        alpha_daily = float(strat_ret.mean() - beta * bench_ret.mean())
        alpha_annual = float(alpha_daily * 252)

    up_mask = bench_ret > 0
    down_mask = bench_ret < 0
    up_capture = _capture_ratio(strat_ret[up_mask], bench_ret[up_mask])
    down_capture = _capture_ratio(strat_ret[down_mask], bench_ret[down_mask])

    return {
        "benchmark_overlap_days": int(len(merged)),
        "benchmark_total_return": bench_total,
        "excess_return": excess_total,
        "information_ratio": info_ratio,
        "beta": beta,
        "alpha_annual": alpha_annual,
        "up_capture": up_capture,
        "down_capture": down_capture,
    }


def _capture_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    if strategy_returns.empty or benchmark_returns.empty:
        return 0.0
    bench_total = float((1 + benchmark_returns).prod() - 1)
    if abs(bench_total) < 1e-12:
        return 0.0
    strat_total = float((1 + strategy_returns).prod() - 1)
    return strat_total / bench_total


def _profit_loss_ratio(sells: pd.DataFrame) -> float:
    if sells.empty or "pnl" not in sells.columns:
        return 0.0
    wins = sells[sells["pnl"] > 0]["pnl"]
    losses = sells[sells["pnl"] < 0]["pnl"]
    if wins.empty or losses.empty:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


def summarize_topk(signals: pd.DataFrame, k: int) -> dict:
    needed = {"candidate", "score", "date"}
    if not needed.issubset(signals.columns):
        return {}
    rows = signals[signals["candidate"]].copy()
    if rows.empty:
        return {
            f"top{k}_candidate_count": 0,
            f"top{k}_precision_start_10d": 0.0,
            f"top{k}_leader_hit_rate_20d": 0.0,
            f"top{k}_avg_future_max_return_20d": 0.0,
            f"top{k}_avg_lead_days": 0.0,
        }
    rows = rows.sort_values(["date", "score"], ascending=[True, False])
    top = rows.groupby("date").head(k)
    out = {f"top{k}_candidate_count": int(len(top))}
    if "y_start_10d" in top.columns:
        out[f"top{k}_precision_start_10d"] = float(top["y_start_10d"].mean())
    if "y_leader_20d" in top.columns:
        out[f"top{k}_leader_hit_rate_20d"] = float(top["y_leader_20d"].mean())
    if "future_max_return_20d" in top.columns:
        out[f"top{k}_avg_future_max_return_20d"] = float(top["future_max_return_20d"].mean())
    if "days_to_start" in top.columns:
        out[f"top{k}_avg_lead_days"] = float(top.loc[top["days_to_start"] > 0, "days_to_start"].mean())
    return out


def write_metrics(metrics: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2, ensure_ascii=False)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if pd.isna(obj):
        return None
    return obj

