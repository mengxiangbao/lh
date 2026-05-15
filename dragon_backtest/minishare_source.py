from __future__ import annotations

import os
from pathlib import Path
from time import sleep

import pandas as pd


SUPPORTED_FREQS = {"5min", "15min", "30min", "60min"}


def init_minishare(token: str | None = None):
    try:
        import minishare as ms
    except ImportError as exc:
        raise ImportError("minishare is not installed. Run: pip install minishare --upgrade") from exc

    token = token or os.getenv("MINISHARE_TOKEN")
    if not token:
        raise ValueError("Missing minishare token. Set $env:MINISHARE_TOKEN or pass --token.")
    return ms.pro_api(token)


def normalize_ts_code(code: str) -> str:
    code = str(code).strip().upper()
    if "." in code:
        return code
    raw = code.zfill(6)
    if raw.startswith(("6", "9")):
        return f"{raw}.SH"
    if raw.startswith(("0", "2", "3")):
        return f"{raw}.SZ"
    if raw.startswith(("4", "8")):
        return f"{raw}.BJ"
    return raw


def read_code_list(codes: str | None = None, codes_file: str | Path | None = None) -> list[str]:
    out: list[str] = []
    if codes:
        out.extend(part.strip() for part in codes.split(",") if part.strip())
    if codes_file:
        path = Path(codes_file)
        if not path.exists():
            raise FileNotFoundError(f"codes file not found: {path}")
        if path.suffix.lower() in {".csv", ".txt"}:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
                col = "ts_code" if "ts_code" in df.columns else "code" if "code" in df.columns else df.columns[0]
                out.extend(df[col].dropna().astype(str).tolist())
            else:
                out.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        else:
            raise ValueError("codes_file must be .csv or .txt")
    normalized = [normalize_ts_code(code) for code in out]
    return sorted(set(normalized))


def fetch_minishare_mins(
    codes: list[str],
    freq: str,
    start: str,
    end: str,
    out_dir: str | Path = "data/raw/minute",
    token: str | None = None,
    sleep_seconds: float = 0.05,
    combine: bool = False,
) -> list[Path]:
    if freq not in SUPPORTED_FREQS:
        raise ValueError(f"Unsupported freq: {freq}. Use one of: {sorted(SUPPORTED_FREQS)}")
    if not codes:
        raise ValueError("No codes provided. Use --codes or --codes-file.")

    pro = init_minishare(token)
    out_dir = Path(out_dir) / freq
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    combined_frames: list[pd.DataFrame] = []
    for i, ts_code in enumerate(codes, start=1):
        print(f"[{i}/{len(codes)}] fetching {ts_code} {freq}")
        df = pro.stk_mins(ts_code=ts_code, freq=freq, start_date=start, end_date=end)
        if df is None or df.empty:
            print(f"  no data: {ts_code}")
            continue
        df = normalize_minute_frame(df, ts_code, freq)
        safe_start = start.replace(":", "").replace("-", "").replace(" ", "_")
        safe_end = end.replace(":", "").replace("-", "").replace(" ", "_")
        path = out_dir / f"{ts_code}_{freq}_{safe_start}_{safe_end}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
        if combine:
            combined_frames.append(df)
        if sleep_seconds > 0:
            sleep(sleep_seconds)

    if combine and combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        safe_start = start.replace(":", "").replace("-", "").replace(" ", "_")
        safe_end = end.replace(":", "").replace("-", "").replace(" ", "_")
        combined_path = out_dir / f"combined_{freq}_{safe_start}_{safe_end}.csv"
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        written.append(combined_path)
    return written


def normalize_minute_frame(df: pd.DataFrame, ts_code: str, freq: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    out["ts_code"] = out.get("ts_code", ts_code)
    out["code"] = out["ts_code"].astype(str).str.split(".").str[0].str.zfill(6)
    out["freq"] = freq

    time_candidates = ["datetime", "trade_time", "trade_datetime", "trade_date", "time"]
    for col in time_candidates:
        if col in out.columns:
            out = out.rename(columns={col: "datetime"})
            break
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        sort_cols = ["ts_code", "datetime"]
    else:
        sort_cols = ["ts_code"]

    preferred = [
        "datetime",
        "ts_code",
        "code",
        "freq",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "volume",
        "amount",
    ]
    ordered = [col for col in preferred if col in out.columns] + [col for col in out.columns if col not in preferred]
    return out[ordered].sort_values(sort_cols).reset_index(drop=True)

