from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_daily
from .features import compute_is_limit_up


def run_regime_analysis(
    daily_path: str | Path,
    backtest_dir: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    backtest_dir = Path(backtest_dir)
    out_dir = Path(out_dir) if out_dir else backtest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    equity = pd.read_csv(backtest_dir / "equity.csv", parse_dates=["date"])
    daily = load_daily(daily_path)
    market = build_market_context(daily)

    tables = {
        "market_context": market,
        "time_slice_performance": time_slice_performance(equity),
        "market_regime_performance": market_regime_performance(equity, market),
        "monthly_contribution": monthly_contribution(equity, market),
    }
    write_regime_tables(tables, out_dir)
    write_regime_summary(tables, out_dir)
    return tables


def build_market_context(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df = df[~df["paused"].astype(bool)].copy()
    df["ret_1d"] = pd.to_numeric(df["close"], errors="coerce") / pd.to_numeric(df["pre_close"], errors="coerce").replace(0, np.nan) - 1
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["ret_x_amount"] = df["ret_1d"].fillna(0.0) * df["amount"]
    df["is_limit_up"] = compute_is_limit_up(df)

    grouped = df.groupby("date", as_index=False).agg(
        market_ret_equal=("ret_1d", "mean"),
        market_amount=("amount", "sum"),
        ret_x_amount=("ret_x_amount", "sum"),
        stock_count=("code", "nunique"),
        positive_ratio=("ret_1d", lambda s: float((s > 0).mean())),
        strong_ratio=("ret_1d", lambda s: float((s > 0.03).mean())),
        weak_ratio=("ret_1d", lambda s: float((s < -0.03).mean())),
        limit_up_count=("is_limit_up", "sum"),
    )
    grouped["market_ret_amount_weighted"] = grouped["ret_x_amount"] / grouped["market_amount"].replace(0, np.nan)
    grouped = grouped.drop(columns=["ret_x_amount"]).sort_values("date").reset_index(drop=True)

    grouped["market_ret_5d"] = rolling_compound(grouped["market_ret_equal"], 5)
    grouped["market_ret_20d"] = rolling_compound(grouped["market_ret_equal"], 20)
    grouped["market_amount_ma20"] = grouped["market_amount"].rolling(20, min_periods=10).mean()
    grouped["market_amount_ma60"] = grouped["market_amount"].rolling(60, min_periods=20).mean()
    grouped["market_amount_ratio"] = grouped["market_amount_ma20"] / grouped["market_amount_ma60"].replace(0, np.nan)
    grouped["positive_ratio_ma5"] = grouped["positive_ratio"].rolling(5, min_periods=3).mean()
    grouped["market_vol_20d"] = grouped["market_ret_equal"].rolling(20, min_periods=10).std()

    grouped["trend_regime"] = np.select(
        [
            grouped["market_ret_20d"] >= 0.05,
            grouped["market_ret_20d"] <= -0.05,
        ],
        ["uptrend", "downtrend"],
        default="range",
    )
    grouped["liquidity_regime"] = np.select(
        [
            grouped["market_amount_ratio"] >= 1.15,
            grouped["market_amount_ratio"] <= 0.85,
        ],
        ["expanding", "shrinking"],
        default="neutral",
    )
    grouped["breadth_regime"] = np.select(
        [
            grouped["positive_ratio_ma5"] >= 0.55,
            grouped["positive_ratio_ma5"] <= 0.45,
        ],
        ["strong_breadth", "weak_breadth"],
        default="neutral_breadth",
    )
    grouped["combined_regime"] = grouped["trend_regime"] + "|" + grouped["liquidity_regime"]
    return grouped


def time_slice_performance(equity: pd.DataFrame) -> pd.DataFrame:
    eq = prepare_equity(equity)
    if eq.empty:
        return pd.DataFrame()

    rows = []
    period_specs = {
        "year": eq["date"].dt.to_period("Y").astype(str),
        "half_year": eq["date"].dt.year.astype(str) + "H" + np.where(eq["date"].dt.month <= 6, "1", "2"),
        "quarter": eq["date"].dt.to_period("Q").astype(str),
        "month": eq["date"].dt.to_period("M").astype(str),
    }
    for period_type, labels in period_specs.items():
        work = eq.assign(period=labels)
        for period, group in work.groupby("period", sort=True):
            rows.append(performance_row(group, period_type, period))
    return pd.DataFrame(rows)


def market_regime_performance(equity: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    eq = prepare_equity(equity)
    merged = eq.merge(market, on="date", how="left")
    if merged.empty:
        return pd.DataFrame()

    rows = []
    for regime_col in ["trend_regime", "liquidity_regime", "breadth_regime", "combined_regime"]:
        for regime, group in merged.dropna(subset=[regime_col]).groupby(regime_col, sort=True):
            row = performance_row(group, regime_col, regime)
            row["market_total_return"] = compound_total(group["market_ret_equal"])
            row["avg_market_return"] = float(group["market_ret_equal"].mean())
            row["avg_positive_ratio"] = float(group["positive_ratio"].mean())
            row["avg_limit_up_count"] = float(group["limit_up_count"].mean())
            row["avg_market_amount_ratio"] = float(group["market_amount_ratio"].mean())
            row["excess_daily_return"] = float((group["daily_return"] - group["market_ret_equal"]).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def monthly_contribution(equity: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    eq = prepare_equity(equity)
    merged = eq.merge(market, on="date", how="left")
    if merged.empty:
        return pd.DataFrame()

    merged["month"] = merged["date"].dt.to_period("M").astype(str)
    rows = []
    for month, group in merged.groupby("month", sort=True):
        row = performance_row(group, "month", month)
        row["market_total_return"] = compound_total(group["market_ret_equal"])
        row["avg_positive_ratio"] = float(group["positive_ratio"].mean())
        row["avg_limit_up_count"] = float(group["limit_up_count"].mean())
        row["dominant_trend_regime"] = most_common(group["trend_regime"])
        row["dominant_liquidity_regime"] = most_common(group["liquidity_regime"])
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    total_abs = out["total_return"].abs().sum()
    out["abs_return_share"] = out["total_return"].abs() / total_abs if total_abs else 0.0
    return out.sort_values("period").reset_index(drop=True)


def prepare_equity(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return equity
    out = equity.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out["daily_return"] = pd.to_numeric(out["daily_return"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def performance_row(group: pd.DataFrame, period_type: str, period: str) -> dict:
    returns = group["daily_return"].fillna(0.0)
    total_return = compound_total(returns)
    return {
        "period_type": period_type,
        "period": str(period),
        "start_date": group["date"].min().date(),
        "end_date": group["date"].max().date(),
        "trading_days": int(len(group)),
        "total_return": float(total_return),
        "annual_return": annualize(total_return, len(group)),
        "max_drawdown": max_drawdown_from_returns(returns),
        "sharpe": sharpe(returns),
        "win_day_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "best_day": float(returns.max()) if len(returns) else 0.0,
        "worst_day": float(returns.min()) if len(returns) else 0.0,
        "avg_positions": float(group.get("positions", pd.Series(dtype=float)).mean()),
    }


def rolling_compound(s: pd.Series, window: int) -> pd.Series:
    return (1 + s.fillna(0.0)).rolling(window, min_periods=max(2, window // 2)).apply(np.prod, raw=True) - 1


def compound_total(s: pd.Series) -> float:
    values = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float((1 + values).prod() - 1)


def annualize(total_return: float, days: int) -> float:
    if days <= 0 or total_return <= -1:
        return -1.0 if total_return <= -1 else 0.0
    return float((1 + total_return) ** (252 / days) - 1)


def max_drawdown_from_returns(returns: pd.Series) -> float:
    wealth = (1 + returns.fillna(0.0)).cumprod()
    if wealth.empty:
        return 0.0
    return float((wealth / wealth.cummax() - 1).min())


def sharpe(returns: pd.Series) -> float:
    returns = returns.fillna(0.0)
    std = returns.std()
    return float(np.sqrt(252) * returns.mean() / std) if std > 0 else 0.0


def most_common(s: pd.Series) -> str:
    values = s.dropna()
    if values.empty:
        return ""
    return str(values.mode().iloc[0])


def write_regime_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def write_regime_summary(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    monthly = tables.get("monthly_contribution", pd.DataFrame())
    regime = tables.get("market_regime_performance", pd.DataFrame())
    summary = {
        "monthly_positive_count": int((monthly.get("total_return", pd.Series(dtype=float)) > 0).sum()) if not monthly.empty else 0,
        "monthly_count": int(len(monthly)),
        "top_months": monthly.sort_values("total_return", ascending=False).head(5).to_dict("records") if not monthly.empty else [],
        "bottom_months": monthly.sort_values("total_return", ascending=True).head(5).to_dict("records") if not monthly.empty else [],
        "best_regimes": regime.sort_values("total_return", ascending=False).head(8).to_dict("records") if not regime.empty else [],
        "worst_regimes": regime.sort_values("total_return", ascending=True).head(8).to_dict("records") if not regime.empty else [],
    }
    with (out_dir / "regime_summary.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2, ensure_ascii=False)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if pd.isna(obj):
        return None
    return obj
