from __future__ import annotations

import hashlib
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


def normalize_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    text = s.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y"})


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


def compute_data_hash(path: str | Path | None = None, df: pd.DataFrame | None = None) -> str:
    resolved = Path(path).resolve() if path else None
    if resolved and resolved.exists():
        return _sha256_file(resolved)
    if df is not None:
        return _hash_dataframe(df)
    raise ValueError("compute_data_hash requires an existing path or a dataframe.")


def convert_table(input_path: str | Path, output_path: str | Path, float32: bool = False) -> Path:
    df = read_table(input_path)
    if float32:
        df = compress_float64_to_float32(df)
    output_path = Path(output_path)
    write_table(df, output_path)
    return output_path


def compress_float64_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        out.loc[:, float_cols] = out[float_cols].astype("float32")
    return out


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
            df[col] = normalize_bool(df[col])
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


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_dataframe(df: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    hasher.update(",".join(map(str, df.columns)).encode("utf-8"))
    hasher.update(str(len(df)).encode("utf-8"))
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        hasher.update(str(dates.min()).encode("utf-8"))
        hasher.update(str(dates.max()).encode("utf-8"))
    sample = pd.concat([df.head(200), df.tail(200)], axis=0)
    hasher.update(sample.to_csv(index=False).encode("utf-8", errors="ignore"))
    return hasher.hexdigest()
