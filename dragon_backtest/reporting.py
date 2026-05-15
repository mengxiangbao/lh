from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MONTH_COLUMNS = [f"{i:02d}" for i in range(1, 13)]


def build_report_tables(result: dict) -> dict[str, pd.DataFrame]:
    equity = result.get("equity", pd.DataFrame()).copy()
    trades = result.get("trades", pd.DataFrame()).copy()
    positions = result.get("positions", pd.DataFrame()).copy()

    tables = {
        "yearly_performance": yearly_performance(equity),
        "monthly_returns": monthly_returns(equity),
        "drawdown_periods": drawdown_periods(equity),
        "trade_distribution": trade_distribution(trades),
        "sector_exposure": sector_exposure(positions),
    }
    return tables


def write_report_tables(result: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in build_report_tables(result).items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def prepare_equity(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return equity
    out = equity.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    out["daily_return"] = out["daily_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def yearly_performance(equity: pd.DataFrame) -> pd.DataFrame:
    equity = prepare_equity(equity)
    if equity.empty:
        return pd.DataFrame()

    rows = []
    for year, group in equity.groupby(equity["date"].dt.year):
        returns = group["daily_return"].fillna(0.0)
        total_return = (1 + returns).prod() - 1
        mdd = max_drawdown_value(group["total_equity"])
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0.0
        rows.append(
            {
                "year": int(year),
                "start_date": group["date"].min().date(),
                "end_date": group["date"].max().date(),
                "trading_days": int(len(group)),
                "total_return": float(total_return),
                "max_drawdown": float(mdd),
                "sharpe": float(sharpe),
                "avg_positions": float(group.get("positions", pd.Series(dtype=float)).mean()),
                "end_equity": float(group["total_equity"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def monthly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    equity = prepare_equity(equity)
    if equity.empty:
        return pd.DataFrame(columns=["year", *MONTH_COLUMNS, "year_total"])

    work = equity.copy()
    work["year"] = work["date"].dt.year
    work["month"] = work["date"].dt.month.map(lambda x: f"{x:02d}")
    monthly = work.groupby(["year", "month"])["daily_return"].apply(lambda s: (1 + s).prod() - 1)
    matrix = monthly.unstack("month").reindex(columns=MONTH_COLUMNS)
    year_total = work.groupby("year")["daily_return"].apply(lambda s: (1 + s).prod() - 1)
    matrix["year_total"] = year_total
    return matrix.reset_index()


def drawdown_periods(equity: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    equity = prepare_equity(equity)
    if equity.empty:
        return pd.DataFrame()

    work = equity[["date", "total_equity"]].copy()
    work = work.reset_index(drop=True)
    work["running_peak"] = work["total_equity"].cummax()
    work["drawdown"] = work["total_equity"] / work["running_peak"] - 1
    in_dd = work["drawdown"] < 0
    groups = (in_dd != in_dd.shift(fill_value=False)).cumsum()
    rows = []
    for _, group in work[in_dd].groupby(groups[in_dd]):
        trough_idx = group["drawdown"].idxmin()
        start_idx = work.loc[: group.index.min(), "total_equity"].idxmax()
        end_candidates = work.loc[group.index.max() + 1 :]
        recovery = end_candidates[end_candidates["drawdown"] >= 0]
        end_date = recovery["date"].iloc[0] if not recovery.empty else pd.NaT
        rows.append(
            {
                "start_date": work.loc[start_idx, "date"].date(),
                "trough_date": work.loc[trough_idx, "date"].date(),
                "recovery_date": end_date.date() if pd.notna(end_date) else "",
                "max_drawdown": float(group["drawdown"].min()),
                "days_to_trough": int((work.loc[trough_idx, "date"] - work.loc[start_idx, "date"]).days),
                "underwater_days": int(
                    ((end_date if pd.notna(end_date) else group["date"].iloc[-1]) - work.loc[start_idx, "date"]).days
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["start_date", "trough_date", "recovery_date", "max_drawdown", "days_to_trough", "underwater_days"])
    return pd.DataFrame(rows).sort_values("max_drawdown").head(top_n).reset_index(drop=True)


def trade_distribution(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "side" not in trades.columns:
        return pd.DataFrame()
    sells = trades[(trades["side"] == "sell") & (trades["status"] == "filled")].copy()
    if sells.empty:
        return pd.DataFrame(columns=["bucket", "count", "avg_return", "total_pnl"])

    sells["return"] = pd.to_numeric(sells["return"], errors="coerce")
    sells["pnl"] = pd.to_numeric(sells["pnl"], errors="coerce").fillna(0.0)
    bins = [-np.inf, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, np.inf]
    labels = ["<=-10%", "-10%~-5%", "-5%~0%", "0%~5%", "5%~10%", "10%~20%", ">20%"]
    sells["bucket"] = pd.cut(sells["return"], bins=bins, labels=labels)
    dist = (
        sells.groupby("bucket", observed=False)
        .agg(count=("return", "size"), avg_return=("return", "mean"), total_pnl=("pnl", "sum"))
        .reset_index()
    )
    return dist


def sector_exposure(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty or "sector" not in positions.columns:
        return pd.DataFrame()
    work = positions.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["market_value"] = pd.to_numeric(work["market_value"], errors="coerce").fillna(0.0)
    daily_total = work.groupby("date")["market_value"].transform("sum")
    work["weight"] = work["market_value"] / daily_total.replace(0, np.nan)
    exposure = (
        work.groupby("sector")
        .agg(
            holding_days=("date", "nunique"),
            avg_weight=("weight", "mean"),
            max_weight=("weight", "max"),
            avg_market_value=("market_value", "mean"),
        )
        .sort_values("avg_weight", ascending=False)
        .reset_index()
    )
    return exposure


def max_drawdown_value(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    return float(drawdown.min()) if not drawdown.empty else 0.0
