from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import load_daily
from .features import prepare_features
from .labels import build_research_labels
from .signals import build_signals


STUDY_FEATURES = [
    "score",
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "rel_ret_5d",
    "rel_ret_20d",
    "ret_20d_rank_in_sector",
    "sector_shift_pct",
    "sector_ret_1d_pct",
    "sector_ret_20d_pct",
    "amount_ma5_to_ma60",
    "up_amount_ratio_20d",
    "obv_slope_20d",
    "cmf_20d",
    "atr20_to_close",
    "range_20d",
    "max_drawdown_20d",
    "close_to_high_60d",
    "close_position_20d",
    "hidden_rs_score",
    "accumulation_score",
    "vol_squeeze_score",
    "anti_fall_score",
    "position_score",
    "limit_up_count_10d",
    "amount",
    "float_mv",
]


@dataclass
class EventStudyConfig:
    pre_window: int = 20
    post_window: int = 20
    min_gap: int = 20
    trigger_volume: float = 1.5
    trigger_ret: float = 0.05
    sector_shift_pct: float = 0.70
    future_return_min: float = 0.25
    future_rank_max: float = 0.10
    fail_return_max: float = 0.10
    fail_rank_min: float = 0.30


def run_event_study(
    data_path: str | Path,
    config_path: str | Path = "config/default.toml",
    out_dir: str | Path = "data/event_study",
    start: str | None = None,
    end: str | None = None,
    study_cfg: EventStudyConfig | None = None,
    save_window: bool = False,
) -> dict[str, pd.DataFrame]:
    study_cfg = study_cfg or EventStudyConfig()
    cfg = load_config(config_path)

    daily = load_daily(data_path, start or None, end or None)
    features = prepare_features(daily)
    labeled = build_research_labels(features)
    signals = build_signals(labeled, cfg, mode="confirmed")
    signals = add_event_flags(signals, study_cfg)
    events = select_distinct_events(signals, study_cfg.min_gap)
    window = build_event_window(signals, events, study_cfg.pre_window, study_cfg.post_window)

    tables = {
        "start_events": events,
        "pre_start_feature_profile": feature_profile(window),
        "positive_vs_negative": positive_vs_negative(window),
        "feature_effects": feature_effects(window),
        "feature_quantiles": feature_quantiles(window),
        "event_samples": event_samples(events),
        "event_summary": event_summary(events),
        "event_rate_by_market": event_rate_by(events, "market"),
        "event_rate_by_sector": event_rate_by(events, "sector"),
    }
    if save_window:
        tables["event_window"] = window

    write_event_study_tables(tables, out_dir)
    return tables


def add_event_flags(df: pd.DataFrame, cfg: EventStudyConfig) -> pd.DataFrame:
    out = df.copy()
    out["event_trigger"] = (
        (out["close"] > out["high_60d_prev"])
        & (out["amount_ma5_to_ma60"] > cfg.trigger_volume)
        & ((out["ret_1d"] > cfg.trigger_ret) | out["is_limit_up"].astype(bool))
        & (out["sector_shift_pct"] >= cfg.sector_shift_pct)
    ).fillna(False)

    positive = (
        out["event_trigger"]
        & (out["future_max_return_20d"] >= cfg.future_return_min)
        & (out["future_rank_in_sector_20d"] <= cfg.future_rank_max)
    )
    negative = (
        out["event_trigger"]
        & (out["future_max_return_20d"] <= cfg.fail_return_max)
        & (out["future_rank_in_sector_20d"] >= cfg.fail_rank_min)
    )
    out["event_class"] = np.select([positive, negative, out["event_trigger"]], ["positive", "negative", "gray"], default="")
    return out


def select_distinct_events(df: pd.DataFrame, min_gap: int) -> pd.DataFrame:
    events = df[df["event_trigger"]].copy()
    if events.empty:
        return pd.DataFrame()

    events = events.sort_values(["code", "date"]).copy()
    events["_bar_index"] = events.groupby("code").cumcount()
    full_bar_index = df[["date", "code"]].copy()
    full_bar_index["_bar_index"] = full_bar_index.groupby("code").cumcount()
    events = events.drop(columns=["_bar_index"], errors="ignore").merge(full_bar_index, on=["date", "code"], how="left")

    keep_indices = []
    last_kept: dict[str, int] = {}
    for idx, row in events.iterrows():
        code = str(row["code"])
        bar_index = int(row["_bar_index"])
        if code not in last_kept or bar_index - last_kept[code] > min_gap:
            keep_indices.append(idx)
            last_kept[code] = bar_index
    events = events.loc[keep_indices].copy()
    events["event_id"] = range(1, len(events) + 1)

    cols = [
        "event_id",
        "date",
        "ts_code",
        "code",
        "name",
        "exchange",
        "market",
        "sector",
        "event_class",
        "score",
        "candidate",
        "pre_pool",
        "trigger",
        "buy_signal",
        "ret_1d",
        "ret_20d",
        "rel_ret_20d",
        "sector_shift_pct",
        "amount_ma5_to_ma60",
        "close_to_high_60d",
        "future_max_return_20d",
        "future_rank_in_sector_20d",
        "days_to_start",
        "_bar_index",
    ]
    cols = [col for col in cols if col in events.columns]
    return events[cols].sort_values(["event_class", "date", "code"]).reset_index(drop=True)


def build_event_window(signals: pd.DataFrame, events: pd.DataFrame, pre_window: int, post_window: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    work = signals.copy().sort_values(["code", "date"]).reset_index(drop=True)
    work["_bar_index"] = work.groupby("code").cumcount()
    by_code = {code: group for code, group in work.groupby("code", sort=False)}
    rows = []

    feature_cols = [col for col in STUDY_FEATURES if col in work.columns]
    id_cols = ["date", "ts_code", "code", "name", "exchange", "market", "sector"]
    id_cols = [col for col in id_cols if col in work.columns]

    for _, event in events.iterrows():
        code = event["code"]
        if code not in by_code:
            continue
        group = by_code[code]
        center = int(event["_bar_index"])
        mask = group["_bar_index"].between(center - pre_window, center + post_window)
        window = group.loc[mask, id_cols + feature_cols].copy()
        if window.empty:
            continue
        window["event_id"] = event["event_id"]
        window["event_date"] = event["date"]
        window["event_class"] = event["event_class"]
        window["future_max_return_20d_at_event"] = event.get("future_max_return_20d", np.nan)
        window["future_rank_in_sector_20d_at_event"] = event.get("future_rank_in_sector_20d", np.nan)
        window["relative_day"] = group.loc[mask, "_bar_index"].to_numpy() - center
        rows.append(window)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["event_id", "relative_day"]).reset_index(drop=True)


def feature_profile(window: pd.DataFrame) -> pd.DataFrame:
    if window.empty:
        return pd.DataFrame()
    feature_cols = [col for col in STUDY_FEATURES if col in window.columns]
    rows = []
    for feature in feature_cols:
        grouped = window.groupby(["event_class", "relative_day"])[feature]
        stat = grouped.agg(
            count="count",
            mean="mean",
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
        ).reset_index()
        stat["feature"] = feature
        rows.append(stat)
    return pd.concat(rows, ignore_index=True)[
        ["event_class", "relative_day", "feature", "count", "mean", "median", "p25", "p75"]
    ]


def positive_vs_negative(window: pd.DataFrame) -> pd.DataFrame:
    if window.empty:
        return pd.DataFrame()
    offsets = [-20, -10, -5, -3, -1, 0]
    feature_cols = [col for col in STUDY_FEATURES if col in window.columns]
    rows = []
    subset = window[window["relative_day"].isin(offsets) & window["event_class"].isin(["positive", "negative"])]
    for rel_day in offsets:
        day = subset[subset["relative_day"] == rel_day]
        for feature in feature_cols:
            pos = day.loc[day["event_class"] == "positive", feature].dropna()
            neg = day.loc[day["event_class"] == "negative", feature].dropna()
            rows.append(
                {
                    "relative_day": rel_day,
                    "feature": feature,
                    "positive_count": int(len(pos)),
                    "negative_count": int(len(neg)),
                    "positive_median": float(pos.median()) if len(pos) else np.nan,
                    "negative_median": float(neg.median()) if len(neg) else np.nan,
                    "median_diff": float(pos.median() - neg.median()) if len(pos) and len(neg) else np.nan,
                    "positive_mean": float(pos.mean()) if len(pos) else np.nan,
                    "negative_mean": float(neg.mean()) if len(neg) else np.nan,
                    "mean_diff": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def feature_effects(window: pd.DataFrame) -> pd.DataFrame:
    if window.empty:
        return pd.DataFrame()
    offsets = [-20, -10, -5, -3, -1, 0]
    feature_cols = [col for col in STUDY_FEATURES if col in window.columns]
    rows = []
    subset = window[window["relative_day"].isin(offsets) & window["event_class"].isin(["positive", "negative"])]
    for rel_day in offsets:
        day = subset[subset["relative_day"] == rel_day]
        for feature in feature_cols:
            pos = day.loc[day["event_class"] == "positive", feature].dropna()
            neg = day.loc[day["event_class"] == "negative", feature].dropna()
            pos_std = pos.std()
            neg_std = neg.std()
            pooled = np.sqrt((pos_std**2 + neg_std**2) / 2) if len(pos) and len(neg) else np.nan
            effect = (pos.mean() - neg.mean()) / pooled if pooled and not np.isnan(pooled) and pooled > 0 else np.nan
            rows.append(
                {
                    "relative_day": rel_day,
                    "feature": feature,
                    "positive_count": int(len(pos)),
                    "negative_count": int(len(neg)),
                    "positive_mean": float(pos.mean()) if len(pos) else np.nan,
                    "negative_mean": float(neg.mean()) if len(neg) else np.nan,
                    "positive_median": float(pos.median()) if len(pos) else np.nan,
                    "negative_median": float(neg.median()) if len(neg) else np.nan,
                    "cohens_d": float(effect) if not np.isnan(effect) else np.nan,
                    "abs_cohens_d": float(abs(effect)) if not np.isnan(effect) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["relative_day", "abs_cohens_d"], ascending=[True, False])


def feature_quantiles(window: pd.DataFrame) -> pd.DataFrame:
    if window.empty:
        return pd.DataFrame()
    offsets = [-20, -10, -5, -3, -1, 0]
    feature_cols = [col for col in STUDY_FEATURES if col in window.columns]
    rows = []
    subset = window[window["relative_day"].isin(offsets)]
    for (event_class, rel_day), group in subset.groupby(["event_class", "relative_day"]):
        for feature in feature_cols:
            values = group[feature].dropna()
            rows.append(
                {
                    "event_class": event_class,
                    "relative_day": rel_day,
                    "feature": feature,
                    "count": int(len(values)),
                    "p10": float(values.quantile(0.10)) if len(values) else np.nan,
                    "p25": float(values.quantile(0.25)) if len(values) else np.nan,
                    "p50": float(values.quantile(0.50)) if len(values) else np.nan,
                    "p75": float(values.quantile(0.75)) if len(values) else np.nan,
                    "p90": float(values.quantile(0.90)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def event_samples(events: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    positive = events[events["event_class"] == "positive"].sort_values("future_max_return_20d", ascending=False).head(top_n)
    negative = events[events["event_class"] == "negative"].sort_values("score", ascending=False).head(top_n)
    gray = events[events["event_class"] == "gray"].sort_values("score", ascending=False).head(top_n)
    return pd.concat([positive, negative, gray], ignore_index=True)


def event_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["event_class", "count", "avg_score", "avg_future_max_return_20d", "avg_future_rank_in_sector_20d"])
    return (
        events.groupby("event_class")
        .agg(
            count=("event_id", "size"),
            avg_score=("score", "mean"),
            avg_future_max_return_20d=("future_max_return_20d", "mean"),
            avg_future_rank_in_sector_20d=("future_rank_in_sector_20d", "mean"),
        )
        .reset_index()
    )


def event_rate_by(events: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if events.empty or group_col not in events.columns:
        return pd.DataFrame()
    pivot = (
        events.pivot_table(index=group_col, columns="event_class", values="event_id", aggfunc="count", fill_value=0)
        .reset_index()
    )
    for col in ["positive", "negative", "gray"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total"] = pivot[["positive", "negative", "gray"]].sum(axis=1)
    pivot["positive_rate"] = pivot["positive"] / pivot["total"].replace(0, np.nan)
    pivot["negative_rate"] = pivot["negative"] / pivot["total"].replace(0, np.nan)
    pivot["pos_neg_ratio"] = pivot["positive"] / pivot["negative"].replace(0, np.nan)
    return pivot.sort_values(["positive_rate", "total"], ascending=[False, False]).reset_index(drop=True)


def write_event_study_tables(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
