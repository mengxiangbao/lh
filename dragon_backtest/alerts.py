from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ALERT_COLUMNS = [
    "date",
    "ts_code",
    "code",
    "name",
    "exchange",
    "market",
    "sector",
    "action",
    "reason",
    "score",
    "candidate_rank",
    "score_pct_all",
    "is_candidate",
    "is_pre_pool",
    "is_trigger",
    "is_buy_signal",
    "market_filter_ok",
    "market_size_mult",
    "sector_shift_pct",
    "hidden_rs_score",
    "accumulation_score",
    "vol_squeeze_score",
    "volatility_energy_score",
    "anti_fall_score",
    "position_score",
    "ret_1d",
    "ret_20d",
    "rel_ret_20d",
    "close_to_high_60d",
    "amount_ma5_to_ma60",
    "up_amount_ratio_20d",
    "close",
    "amount",
    "float_mv",
    "limit_up_count_10d",
    "market_trend_regime",
    "market_liquidity_regime",
    "market_breadth_regime",
    "market_positive_ratio_ma5",
]


def build_alerts(signals: pd.DataFrame, top_n: int = 30, include_watch: bool = True) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    df = signals.copy()
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.split(".").str[0].str.zfill(6)
    if "ts_code" not in df.columns and "code" in df.columns:
        df["ts_code"] = df["code"]
    for col in ["name", "exchange", "market"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["candidate", "pre_pool", "trigger", "buy_signal"]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)

    df["action"] = np.select(
        [
            df["buy_signal"],
            df["pre_pool"] & df["trigger"],
            df["candidate"] & (df.get("amount_ma5_to_ma60", 0) >= 3),
            df["candidate"] & (df.get("sector_shift_pct", 1) < 0.50),
            df["candidate"],
        ],
        [
            "次日可买",
            "次日可买",
            "过热风险",
            "转弱剔除",
            "观察",
        ],
        default="",
    )
    df["reason"] = df.apply(alert_reason, axis=1)
    df["is_candidate"] = df["candidate"]
    df["is_pre_pool"] = df["pre_pool"]
    df["is_trigger"] = df["trigger"]
    df["is_buy_signal"] = df["buy_signal"]
    priority = {"次日可买": 0, "观察": 1, "过热风险": 2, "转弱剔除": 3, "": 9}
    df["action_priority"] = df["action"].map(priority).fillna(9)

    rows = df[df["action"].ne("")].copy()
    if not include_watch:
        rows = rows[rows["action"].ne("观察")]

    rows = rows.sort_values(["date", "action_priority", "score"], ascending=[True, True, False])
    selected = rows.groupby("date", group_keys=False).head(top_n)
    cols = [col for col in ALERT_COLUMNS if col in selected.columns]
    return selected[cols].reset_index(drop=True)


def alert_reason(row: pd.Series) -> str:
    if bool(row.get("buy_signal", False)):
        parts = ["潜质池", "启动确认"]
        if bool(row.get("trigger_breakout_60d", False)):
            parts.append("突破60日高点")
        if bool(row.get("trigger_volume", False)):
            parts.append("放量")
        if bool(row.get("trigger_strong_day", False)):
            parts.append("强势日")
        if bool(row.get("trigger_sector", False)):
            parts.append("板块共振")
        return " / ".join(parts)

    if bool(row.get("candidate", False)) and row.get("amount_ma5_to_ma60", 0) >= 3:
        return "成交额短期放大过高，可能已过热"

    if bool(row.get("candidate", False)) and row.get("sector_shift_pct", 1) < 0.50:
        return "板块边际强度转弱"

    if bool(row.get("candidate", False)):
        parts = ["潜质分靠前"]
        if row.get("sector_shift_pct", 0) >= 0.70:
            parts.append("板块改善")
        if row.get("rel_ret_20d", 0) > 0:
            parts.append("强于板块")
        if row.get("close_to_high_60d", 0) >= 0.90:
            parts.append("接近60日高点")
        if row.get("up_amount_ratio_20d", 0) >= 1.2:
            parts.append("温和吸筹")
        return " / ".join(parts)

    return ""


def write_alerts(alerts: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(out_dir / "alerts.csv", index=False, encoding="utf-8-sig")
    if alerts.empty:
        return
    latest_date = pd.to_datetime(alerts["date"]).max()
    latest = alerts[pd.to_datetime(alerts["date"]) == latest_date]
    latest.to_csv(out_dir / "alerts_latest.csv", index=False, encoding="utf-8-sig")
