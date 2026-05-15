from __future__ import annotations

from pathlib import Path

import pandas as pd


def fix_daily_data(input_path: str | Path, output_path: str | Path) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = pd.read_csv(input_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.split(".").str[0].str.zfill(6)

    first_trade = df.groupby("code")["date"].transform("min")
    missing_list_date = df["list_date"].isna()
    list_after_first_trade = df["list_date"] > first_trade
    fixed_list_date_rows = missing_list_date | list_after_first_trade
    df.loc[fixed_list_date_rows, "list_date"] = first_trade.loc[fixed_list_date_rows]

    if "sector" in df.columns:
        missing_sector = df["sector"].isna() | df["sector"].astype(str).str.strip().isin(["", "nan", "None", "unknown"])
        df.loc[missing_sector, "sector"] = "unknown"
    else:
        missing_sector = pd.Series(False, index=df.index)

    for col in ["paused", "is_st"]:
        if col in df.columns:
            df[col] = normalize_bool(df[col])

    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(len(df)),
        "stock_count": int(df["code"].nunique()),
        "start_date": str(df["date"].min().date()),
        "end_date": str(df["date"].max().date()),
        "fixed_list_date_rows": int(fixed_list_date_rows.sum()),
        "fixed_sector_rows": int(missing_sector.sum()) if "sector" in df.columns else 0,
    }


def normalize_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    text = s.fillna(False).astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y"})
