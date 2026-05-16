from __future__ import annotations

from pathlib import Path
from time import sleep
from typing import Callable

import pandas as pd

from .data import normalize_bool, write_table
from .tushare_client import init_tushare


def yyyymmdd(value: str) -> str:
    return str(value).replace("-", "")[:8]


def ts_code_to_code(ts_code: pd.Series) -> pd.Series:
    return ts_code.astype(str).str.split(".").str[0].str.zfill(6)


def fetch_tushare_daily(
    start: str,
    end: str,
    out_path: str | Path = "data/raw/daily_price.csv",
    token: str | None = None,
    http_url: str | None = None,
    sleep_seconds: float = 0.12,
    skip_namechange: bool = False,
    cache_dir: str | Path | None = None,
    retries: int = 3,
    retry_sleep: float = 2.0,
) -> Path:
    pro = init_tushare(token, http_url)
    start_date = yyyymmdd(start)
    end_date = yyyymmdd(end)

    basics = fetch_stock_basic(pro)
    trade_dates = fetch_trade_dates(pro, start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No open trade dates found from {start_date} to {end_date}")

    daily_frames = []
    for i, trade_date in enumerate(trade_dates, start=1):
        print(f"[{i}/{len(trade_dates)}] fetching {trade_date}")
        one_day = fetch_one_trade_date_cached(
            pro=pro,
            trade_date=trade_date,
            cache_dir=cache_dir,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        if not one_day.empty:
            daily_frames.append(one_day)
        if sleep_seconds > 0:
            sleep(sleep_seconds)

    if not daily_frames:
        raise RuntimeError("No daily market data returned from Tushare.")

    daily = pd.concat(daily_frames, ignore_index=True)
    daily = merge_stock_basic(daily, basics)

    if skip_namechange:
        daily["is_st"] = daily["name"].fillna("").str.contains("ST", case=False, regex=False)
    else:
        namechange = fetch_namechange(pro, basics["ts_code"].dropna().unique(), sleep_seconds)
        daily = apply_namechange_st(daily, namechange)

    daily = finalize_daily_schema(daily)
    out_path = Path(out_path)
    write_table(daily, out_path)
    return out_path


def fetch_stock_basic(pro) -> pd.DataFrame:
    frames = []
    fields = "ts_code,symbol,name,area,industry,market,exchange,list_date,delist_date"
    for status in ["L", "D", "P"]:
        try:
            part = pro.stock_basic(exchange="", list_status=status, fields=fields)
        except Exception:
            part = pro.stock_basic(exchange="", list_status=status)
        if part is not None and not part.empty:
            frames.append(part)
    if not frames:
        raise RuntimeError("stock_basic returned no rows.")
    basics = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")
    if "delist_date" not in basics.columns:
        basics["delist_date"] = ""
    return basics


def fetch_trade_dates(pro, start_date: str, end_date: str) -> list[str]:
    cal = pro.trade_cal(exchange="", start_date=start_date, end_date=end_date, is_open="1")
    if cal is None or cal.empty:
        return []
    return sorted(cal["cal_date"].astype(str).tolist())


def fetch_one_trade_date_cached(
    pro,
    trade_date: str,
    cache_dir: str | Path | None,
    retries: int,
    retry_sleep: float,
) -> pd.DataFrame:
    if cache_dir:
        cache_path = Path(cache_dir) / f"{trade_date}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path, dtype={"trade_date": str})

    out = fetch_one_trade_date(pro, trade_date, retries=retries, retry_sleep=retry_sleep)
    if cache_dir and not out.empty:
        cache_path = Path(cache_dir) / f"{trade_date}.csv"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return out


def fetch_one_trade_date(pro, trade_date: str, retries: int = 3, retry_sleep: float = 2.0) -> pd.DataFrame:
    daily = call_with_retry(
        lambda: pro.daily(
            trade_date=trade_date,
            fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
        ),
        name=f"daily {trade_date}",
        retries=retries,
        retry_sleep=retry_sleep,
    )
    if daily is None or daily.empty:
        return pd.DataFrame()

    basic = call_with_retry(
        lambda: pro.daily_basic(
            trade_date=trade_date,
            fields="ts_code,trade_date,turnover_rate,total_mv,circ_mv",
        ),
        name=f"daily_basic {trade_date}",
        retries=retries,
        retry_sleep=retry_sleep,
    )
    limit = call_with_retry(
        lambda: pro.stk_limit(
            trade_date=trade_date,
            fields="ts_code,trade_date,up_limit,down_limit",
        ),
        name=f"stk_limit {trade_date}",
        retries=retries,
        retry_sleep=retry_sleep,
    )

    out = daily.merge(basic, on=["ts_code", "trade_date"], how="left")
    out = out.merge(limit, on=["ts_code", "trade_date"], how="left")
    out["paused"] = False
    return out


def call_with_retry(func: Callable[[], pd.DataFrame], name: str, retries: int, retry_sleep: float):
    last_exc = None
    for attempt in range(1, max(int(retries), 1) + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt >= max(int(retries), 1):
                break
            print(f"{name} failed on attempt {attempt}/{retries}: {exc}; retrying in {retry_sleep}s")
            if retry_sleep > 0:
                sleep(retry_sleep)
    raise last_exc


def merge_stock_basic(daily: pd.DataFrame, basics: pd.DataFrame) -> pd.DataFrame:
    keep = ["ts_code", "symbol", "name", "industry", "market", "exchange", "list_date", "delist_date"]
    keep = [col for col in keep if col in basics.columns]
    out = daily.merge(basics[keep], on="ts_code", how="left")
    out["sector"] = out.get("industry", pd.Series(index=out.index, dtype=object)).fillna("unknown")
    return out


def fetch_namechange(pro, ts_codes, sleep_seconds: float) -> pd.DataFrame:
    frames = []
    codes = list(ts_codes)
    for i, ts_code in enumerate(codes, start=1):
        if i % 100 == 0:
            print(f"fetching namechange {i}/{len(codes)}")
        try:
            part = pro.namechange(
                ts_code=ts_code,
                fields="ts_code,name,start_date,end_date,change_reason",
            )
        except Exception:
            part = pd.DataFrame()
        if part is not None and not part.empty:
            frames.append(part)
        if sleep_seconds > 0:
            sleep(sleep_seconds)
    if not frames:
        return pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date", "change_reason"])
    return pd.concat(frames, ignore_index=True)


def apply_namechange_st(daily: pd.DataFrame, namechange: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["is_st"] = False
    if namechange.empty:
        out["is_st"] = out["name"].fillna("").str.contains("ST", case=False, regex=False)
        return out

    out["_trade_date"] = pd.to_datetime(out["trade_date"])
    namechange = namechange.copy()
    namechange["start_date"] = pd.to_datetime(namechange["start_date"], errors="coerce")
    namechange["end_date"] = pd.to_datetime(namechange["end_date"], errors="coerce").fillna(pd.Timestamp.max)
    st_intervals = namechange[
        namechange["name"].fillna("").str.contains("ST", case=False, regex=False)
        | namechange["change_reason"].fillna("").str.contains("ST", case=False, regex=False)
    ]
    for ts_code, intervals in st_intervals.groupby("ts_code"):
        code_mask = out["ts_code"].eq(ts_code)
        if not code_mask.any():
            continue
        dates = out.loc[code_mask, "_trade_date"]
        st_mask = pd.Series(False, index=dates.index)
        for _, interval in intervals.iterrows():
            st_mask |= (dates >= interval["start_date"]) & (dates <= interval["end_date"])
        out.loc[st_mask.index, "is_st"] = out.loc[st_mask.index, "is_st"] | st_mask
    return out.drop(columns=["_trade_date"])


def finalize_daily_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["trade_date"])
    out["code"] = ts_code_to_code(df["ts_code"])
    out["ts_code"] = df["ts_code"].astype(str)
    out["name"] = df.get("name", pd.Series(index=df.index, dtype=object)).fillna("")
    out["exchange"] = df.get("exchange", pd.Series(index=df.index, dtype=object)).fillna("")
    out["market"] = df.get("market", pd.Series(index=df.index, dtype=object)).fillna("")
    out["open"] = df["open"].astype(float)
    out["high"] = df["high"].astype(float)
    out["low"] = df["low"].astype(float)
    out["close"] = df["close"].astype(float)
    out["pre_close"] = df["pre_close"].astype(float)
    out["volume"] = df["vol"].fillna(0).astype(float) * 100.0
    out["amount"] = df["amount"].fillna(0).astype(float) * 1000.0
    out["up_limit"] = df["up_limit"].fillna(out["pre_close"] * 1.10).astype(float)
    out["down_limit"] = df["down_limit"].fillna(out["pre_close"] * 0.90).astype(float)
    out["paused"] = normalize_bool(df["paused"])
    out["is_st"] = normalize_bool(df["is_st"])
    out["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    out["sector"] = df["sector"].fillna("unknown").astype(str)
    out["float_mv"] = df["circ_mv"].fillna(0).astype(float) * 10000.0
    out["total_mv"] = df["total_mv"].fillna(0).astype(float) * 10000.0
    out["turnover"] = df["turnover_rate"].fillna(0).astype(float) / 100.0

    out = out.dropna(subset=["date", "code", "open", "high", "low", "close", "pre_close"])
    out = out.sort_values(["code", "date"]).reset_index(drop=True)
    return out
