from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import REQUIRED_DAILY_COLUMNS


@dataclass
class DataIssue:
    severity: str
    check: str
    message: str
    rows: int = 0
    sample: list[dict[str, Any]] | None = None


def check_daily_data(
    path: str | Path,
    out_dir: str | Path | None = None,
    sample_rows: int = 10,
) -> dict:
    path = Path(path)
    df = read_raw_table(path)
    issues: list[DataIssue] = []

    missing = sorted(REQUIRED_DAILY_COLUMNS.difference(df.columns))
    if missing:
        issues.append(
            DataIssue(
                severity="error",
                check="schema.required_columns",
                message=f"缺少必需字段: {', '.join(missing)}",
                rows=0,
            )
        )
        return write_report(path, df, issues, out_dir)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)

    check_basic_shape(df, issues, sample_rows)
    check_duplicates(df, issues, sample_rows)
    check_missing_values(df, issues, sample_rows)
    check_ohlc(df, issues, sample_rows)
    check_limits(df, issues, sample_rows)
    check_suspension(df, issues, sample_rows)
    check_listing_dates(df, issues, sample_rows)
    check_sector(df, issues, sample_rows)
    check_amount_units(df, issues, sample_rows)
    check_market_values(df, issues, sample_rows)
    check_boolean_columns(df, issues, sample_rows)

    return write_report(path, df, issues, out_dir)


def read_raw_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def check_basic_shape(df: pd.DataFrame, issues: list[DataIssue], sample_rows: int) -> None:
    if df.empty:
        issues.append(DataIssue("error", "shape.empty", "数据为空，无法回测。"))
        return
    invalid_date = df["date"].isna()
    if invalid_date.any():
        add_issue(issues, "error", "date.invalid", "存在无法解析的交易日期。", df[invalid_date], sample_rows)
    code_text = df["code"].astype(str).str.strip().str.lower()
    code_missing = df["code"].isna() | code_text.isin(["", "nan", "none", "000nan", "00none"])
    if code_missing.any():
        add_issue(issues, "error", "code.missing", "存在缺失股票代码的记录。", df[code_missing], sample_rows)


def check_duplicates(df: pd.DataFrame, issues: list[DataIssue], sample_rows: int) -> None:
    dup = df.duplicated(["date", "code"], keep=False)
    if dup.any():
        add_issue(issues, "error", "date_code.duplicate", "存在 date + code 重复记录。", df[dup], sample_rows)


def check_missing_values(df: pd.DataFrame, issues: list[DataIssue], sample_rows: int) -> None:
    critical = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "up_limit",
        "down_limit",
        "paused",
        "is_st",
        "list_date",
        "sector",
    ]
    for col in critical:
        mask = df[col].isna()
        if mask.any():
            severity = "error" if col in {"open", "high", "low", "close", "pre_close"} else "warning"
            add_issue(issues, severity, f"{col}.missing", f"`{col}` 存在缺失值。", df[mask], sample_rows)


def check_ohlc(df: pd.DataFrame, issues: list[DataIssue], sample_rows: int) -> None:
    price_cols = ["open", "high", "low", "close", "pre_close"]
    non_positive = df[price_cols].le(0).any(axis=1)
    if non_positive.any():
        add_issue(issues, "error", "price.non_positive", "价格字段存在小于等于 0 的记录。", df[non_positive], sample_rows)

    bad_high_low = df["high"] < df["low"]
    if bad_high_low.any():
        add_issue(issues, "error", "ohlc.high_lt_low", "存在 high < low 的记录。", df[bad_high_low], sample_rows)

    open_out = (df["open"] > df["high"] * 1.0001) | (df["open"] < df["low"] * 0.9999)
    close_out = (df["close"] > df["high"] * 1.0001) | (df["close"] < df["low"] * 0.9999)
    if open_out.any():
        add_issue(issues, "error", "ohlc.open_out_of_range", "存在 open 不在 [low, high] 区间内的记录。", df[open_out], sample_rows)
    if close_out.any():
        add_issue(issues, "error", "ohlc.close_out_of_range", "存在 close 不在 [low, high] 区间内的记录。", df[close_out], sample_rows)


def check_limits(df: pd.DataFrame, issues: list[DataIssue], sample_rows: int) -> None:
    no_limit = is_no_price_limit(df)
    bad_limit = ((df["up_limit"] <= 0) | (df["down_limit"] <= 0) | (df["up_limit"] <= df["down_limit"])) & ~no_limit
    if bad_limit.any():
        add_issue(issues, "error", "limit.invalid", "涨跌停价格异常：up_limit/down_limit 小于等于 0 或 up_limit <= down_limit。", df[bad_limit], sample_rows)

    if no_limit.any():
        add_issue(
            issues,
            "info",
            "limit.no_price_limit_sentinel",
            "存在无涨跌幅限制哨兵值，例如新股上市初期；撮合时不会按涨跌停阻断。",
            df[no_limit],
            sample_rows,
        )

    close_above = (df["close"] > df["up_limit"] * 1.001) & ~no_limit
    close_below = (df["close"] < df["down_limit"] * 0.999) & ~no_limit
    if close_above.any():
        add_issue(issues, "warning", "limit.close_above_up_limit", "存在 close 高于 up_limit 的记录，请确认新股/无涨跌幅限制口径。", df[close_above], sample_rows)
    if close_below.any():
        add_issue(issues, "warning", "limit.close_below_down_limit", "存在 close 低于 down_limit 的记录，请确认新股/无涨跌幅限制口径。", df[close_below], sample_rows)

    up_ratio = df["up_limit"] / df["pre_close"].replace(0, np.nan) - 1
    down_ratio = df["down_limit"] / df["pre_close"].replace(0, np.nan) - 1
    odd_ratio = (up_ratio.gt(0.35) | up_ratio.lt(0.03) | down_ratio.gt(-0.03) | down_ratio.lt(-0.35)) & ~no_limit
    if odd_ratio.any():
        add_issue(
            issues,
            "info",
            "limit.ratio_unusual",
            "部分涨跌停比例不在常见范围内，可能是新股、北交所、复权口径不一致或数据异常。",
            df[odd_ratio],
            sample_rows,
        )


def is_no_price_limit(df: pd.DataFrame) -> pd.Series:
    up = pd.to_numeric(df["up_limit"], errors="coerce")
    down = pd.to_numeric(df["down_limit"], errors="coerce")
    pre_close = pd.to_numeric(df["pre_close"], errors="coerce").replace(0, np.nan)
    up_ratio = up / pre_close - 1
    return (up >= 9999) | (down <= 0.01) | (up_ratio > 5)


def check_suspension(df: pd.DataFrame, issues: list[DataIssue], sample_rows: int) -> None:
    paused = coerce_bool(df["paused"])
    paused_with_volume = paused & ((df["volume"] > 0) | (df["amount"] > 0))
    if paused_with_volume.any():
        add_issue(issues, "warning", "paused.has_volume", "停牌记录存在成交量或成交额，请确认 paused 口径。", df[paused_with_volume], sample_rows)

    active_no_volume = (~paused) & ((df["volume"] <= 0) | (df["amount"] <= 0))
    if active_no_volume.any():
        add_issue(issues, "warning", "active.no_volume", "非停牌记录存在 volume 或 amount 小于等于 0。", df[active_no_volume], sample_rows)


def check_listing_dates(df: pd.DataFrame, issues: list[DataIssue], sample_rows: int) -> None:
    bad_list_date = df["list_date"].isna()
    if bad_list_date.any():
        add_issue(issues, "warning", "list_date.missing_or_invalid", "上市日期缺失或无法解析。", df[bad_list_date], sample_rows)
    trade_before_list = df["date"] < df["list_date"]
    if trade_before_list.any():
        add_issue(issues, "error", "list_date.after_trade_date", "存在交易日期早于上市日期的记录。", df[trade_before_list], sample_rows)


def check_sector(df: pd.DataFrame, issues: list[DataIssue], sample_rows: int) -> None:
    missing_sector = df["sector"].isna() | df["sector"].astype(str).str.strip().isin(["", "nan", "None", "unknown"])
    missing_rate = float(missing_sector.mean()) if len(df) else 0.0
    if missing_rate > 0.20:
        add_issue(issues, "warning", "sector.high_missing_rate", f"行业/板块字段缺失率较高：{missing_rate:.2%}。", df[missing_sector], sample_rows)
    elif missing_sector.any():
        add_issue(issues, "info", "sector.some_missing", f"行业/板块字段存在少量缺失：{missing_rate:.2%}。", df[missing_sector], sample_rows)

    sector_counts = df.groupby("date")["sector"].nunique(dropna=True)
    if not sector_counts.empty and sector_counts.median() <= 2:
        issues.append(
            DataIssue(
                severity="warning",
                check="sector.too_few_daily_groups",
                message="多数交易日行业/板块数量过少，板块强度因子可能失真。",
                rows=int(len(sector_counts)),
            )
        )


def check_amount_units(df: pd.DataFrame, issues: list[DataIssue], sample_rows: int) -> None:
    positive_amount = df.loc[df["amount"] > 0, "amount"]
    if positive_amount.empty:
        issues.append(DataIssue("error", "amount.empty_positive", "成交额全部为空或小于等于 0。"))
        return

    median_amount = float(positive_amount.median())
    if median_amount < 100_000:
        issues.append(
            DataIssue(
                severity="warning",
                check="amount.unit_suspicious_small",
                message=f"成交额中位数为 {median_amount:.2f}，看起来不像人民币元口径；Tushare daily.amount 通常需要乘以 1000。",
            )
        )
    if median_amount > 200_000_000_000:
        issues.append(
            DataIssue(
                severity="warning",
                check="amount.unit_suspicious_large",
                message=f"成交额中位数为 {median_amount:.2f}，看起来过大，请确认单位。",
            )
        )


def check_market_values(df: pd.DataFrame, issues: list[DataIssue], sample_rows: int) -> None:
    bad_mv = (df["float_mv"] < 0) | (df["total_mv"] < 0)
    if bad_mv.any():
        add_issue(issues, "warning", "market_value.negative", "市值字段存在负数。", df[bad_mv], sample_rows)
    float_gt_total = (df["float_mv"] > df["total_mv"] * 1.05) & (df["total_mv"] > 0)
    if float_gt_total.any():
        add_issue(issues, "warning", "market_value.float_gt_total", "流通市值显著大于总市值，请确认单位或字段映射。", df[float_gt_total], sample_rows)


def check_boolean_columns(df: pd.DataFrame, issues: list[DataIssue], sample_rows: int) -> None:
    for col in ["paused", "is_st"]:
        raw = df[col].dropna().astype(str).str.strip().str.lower()
        allowed = {"0", "1", "true", "false", "yes", "no", "y", "n"}
        bad = ~raw.isin(allowed)
        if bad.any():
            bad_index = raw[bad].index
            add_issue(issues, "warning", f"{col}.non_boolean_values", f"`{col}` 存在非布尔风格取值。", df.loc[bad_index], sample_rows)


def coerce_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    text = s.fillna(False).astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y"})


def add_issue(
    issues: list[DataIssue],
    severity: str,
    check: str,
    message: str,
    rows: pd.DataFrame,
    sample_rows: int,
) -> None:
    issues.append(
        DataIssue(
            severity=severity,
            check=check,
            message=message,
            rows=int(len(rows)),
            sample=sample_records(rows, sample_rows),
        )
    )


def sample_records(df: pd.DataFrame, sample_rows: int) -> list[dict[str, Any]]:
    cols = [col for col in ["date", "code", "sector", "open", "high", "low", "close", "pre_close", "amount", "up_limit", "down_limit", "paused", "is_st"] if col in df.columns]
    sample = df[cols].head(sample_rows).copy()
    for col in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[col]):
            sample[col] = sample[col].dt.strftime("%Y-%m-%d")
    return sample.replace({np.nan: None}).to_dict(orient="records")


def write_report(path: Path, df: pd.DataFrame, issues: list[DataIssue], out_dir: str | Path | None) -> dict:
    severity_counts = {
        "error": sum(1 for issue in issues if issue.severity == "error"),
        "warning": sum(1 for issue in issues if issue.severity == "warning"),
        "info": sum(1 for issue in issues if issue.severity == "info"),
    }
    report = {
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "optional_columns_present": [col for col in ["ts_code", "name", "exchange", "market"] if col in df.columns],
        "stock_count": int(df["code"].nunique()) if "code" in df.columns else 0,
        "start_date": str(pd.to_datetime(df["date"]).min().date()) if "date" in df.columns and len(df) else None,
        "end_date": str(pd.to_datetime(df["date"]).max().date()) if "date" in df.columns and len(df) else None,
        "severity_counts": severity_counts,
        "can_backtest": severity_counts["error"] == 0,
        "issues": [asdict(issue) for issue in issues],
    }

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.Series(report).drop(labels=["issues", "columns"], errors="ignore").to_json(
            out_dir / "data_check_summary.json",
            force_ascii=False,
            indent=2,
        )
        pd.DataFrame([asdict(issue) | {"sample": ""} for issue in issues]).to_csv(
            out_dir / "data_check_issues.csv",
            index=False,
            encoding="utf-8-sig",
        )
        write_json_report(report, out_dir / "data_check_report.json")
    return report


def write_json_report(report: dict, path: Path) -> None:
    import json

    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
