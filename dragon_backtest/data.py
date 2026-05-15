from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_DAILY_COLUMNS = {
    "date",
    "code",
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
    "float_mv",
    "total_mv",
    "turnover",
}

OPTIONAL_DAILY_COLUMNS = {
    "ts_code",
    "name",
    "exchange",
    "market",
}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")
    return normalize_daily_columns(df) if "code" in df.columns else df


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix in {".csv", ".txt"}:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")


def normalize_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["date", "list_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str).str.upper()
    if "name" in df.columns:
        df["name"] = df["name"].astype(str)
    for col in ["exchange", "market"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "sector" in df.columns:
        df["sector"] = df["sector"].astype(str)
    for col in ["paused", "is_st"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    return df


def load_daily(path: str | Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    df = read_table(path)
    missing = REQUIRED_DAILY_COLUMNS.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Daily data missing required columns: {missing_text}")
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df
