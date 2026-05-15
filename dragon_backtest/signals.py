from __future__ import annotations

import numpy as np
import pandas as pd


def add_rule_score(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = features.copy()
    weights = cfg["score"]["weights"]
    weighted = (
        weights.get("hidden_rs", 0.0) * df["hidden_rs_score"]
        + weights.get("accumulation", 0.0) * df["accumulation_score"]
        + weights.get("vol_squeeze", 0.0) * df["vol_squeeze_score"]
        + weights.get("volatility_energy", 0.0) * df.get("volatility_energy_score", 0.0)
        + weights.get("anti_fall", 0.0) * df["anti_fall_score"]
        + weights.get("sector_shift", 0.0) * df["sector_shift_pct"]
        + weights.get("position", 0.0) * df["position_score"]
        + weights.get("limit_gene", 0.0) * df["limit_gene_score"]
    )
    denom = sum(abs(v) for v in weights.values()) or 1.0
    df["score"] = (weighted / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1)
    df["score_pct_all"] = df.groupby("date")["score"].rank(pct=True, ascending=True)
    return df


def build_signals(features: pd.DataFrame, cfg: dict, mode: str | None = None) -> pd.DataFrame:
    df = add_rule_score(features, cfg)
    universe_cfg = cfg["universe"]
    signal_cfg = cfg["signal"]
    mode = mode or signal_cfg.get("mode", "confirmed")

    candidate = pd.Series(True, index=df.index)
    candidate &= df["list_days"] > universe_cfg["min_list_days"]
    if universe_cfg.get("exclude_st", True):
        candidate &= ~df["is_st"].astype(bool)
    if universe_cfg.get("exclude_suspended", True):
        candidate &= ~df["paused"].astype(bool)
    candidate &= df["amount_ma20"] > universe_cfg["min_avg_amount_20d"]
    candidate &= df["ret_20d"] < universe_cfg["max_ret_20d"]
    candidate &= df["limit_up_count_10d"] <= universe_cfg["max_limit_up_count_10d"]
    candidate &= df["amount_ma5_to_ma60"] < universe_cfg["max_amount_ma5_to_ma60"]
    candidate &= df["close_to_high_60d"] > universe_cfg["min_close_to_high_60d"]
    candidate &= df["sector_shift_pct"] >= universe_cfg["min_sector_shift_pct"]
    candidate &= df["up_amount_ratio_20d"] >= universe_cfg["min_up_amount_ratio_20d"]

    if universe_cfg.get("require_stock_stronger_than_sector", True):
        candidate &= df["rel_ret_20d"] > 0
        candidate &= df["ret_20d_rank_in_sector"] <= 0.20
    if universe_cfg.get("max_drawdown_vs_sector_median", True):
        candidate &= df["max_drawdown_20d"] >= df["sector_max_drawdown_20d_median"]

    df["candidate"] = candidate.fillna(False)
    df["candidate_rank"] = np.nan
    candidate_rows = df["candidate"]
    df.loc[candidate_rows, "candidate_rank"] = df.loc[candidate_rows].groupby("date")["score"].rank(
        method="first", ascending=False
    )

    candidate_top_n = signal_cfg["candidate_top_n"]
    top_pct_cutoff = 1.0 - signal_cfg["candidate_top_pct"]
    df["pre_pool"] = df["candidate"] & (
        (df["candidate_rank"] <= candidate_top_n) | (df["score_pct_all"] >= top_pct_cutoff)
    )

    in_day_location = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["trigger_breakout_60d"] = df["close"] > df["high_60d_prev"]
    df["trigger_volume"] = (df["amount"] / df["amount_ma60"].replace(0, np.nan)) > signal_cfg["trigger_amount_to_ma60"]
    df["trigger_strong_day"] = (df["ret_1d"] > signal_cfg["trigger_ret_1d"]) | df["is_limit_up"].astype(bool)
    df["trigger_sector"] = df["sector_ret_1d_pct"] >= signal_cfg["trigger_sector_ret_pct"]
    df["trigger_close_location"] = in_day_location >= signal_cfg["trigger_close_location"]
    df["trigger"] = (
        df["trigger_breakout_60d"]
        & df["trigger_volume"]
        & df["trigger_strong_day"]
        & df["trigger_sector"]
        & df["trigger_close_location"]
    ).fillna(False)
    df["market_filter_ok"] = build_market_filter(df, cfg)

    direct_top_n = signal_cfg["direct_top_n"]
    if mode == "potential":
        df["buy_signal"] = df["candidate"] & (df["candidate_rank"] <= direct_top_n)
        df["target_weight_mult"] = 1.0
    elif mode == "confirmed":
        df["buy_signal"] = df["pre_pool"] & df["trigger"]
        df["target_weight_mult"] = 1.0
    elif mode == "hybrid":
        early = df["candidate"] & (df["candidate_rank"] <= direct_top_n)
        confirmed = df["pre_pool"] & df["trigger"]
        df["buy_signal"] = early | confirmed
        df["target_weight_mult"] = np.where(confirmed, 1.0, 0.5)
    else:
        raise ValueError(f"Unknown signal mode: {mode}")

    df["market_size_mult"] = build_market_size_mult(df, cfg)
    df["target_weight_mult"] = (df["target_weight_mult"].astype(float) * df["market_size_mult"]).clip(lower=0.0)

    if cfg.get("market_filter", {}).get("enabled", False) and cfg.get("market_filter", {}).get("apply_to", "buy") == "buy":
        df["raw_buy_signal"] = df["buy_signal"]
        df["buy_signal"] = df["buy_signal"] & df["market_filter_ok"]
    else:
        df["raw_buy_signal"] = df["buy_signal"]

    df["signal_mode"] = mode
    return df.sort_values(["date", "code"]).reset_index(drop=True)


def build_market_filter(df: pd.DataFrame, cfg: dict) -> pd.Series:
    market_cfg = cfg.get("market_filter", {})
    ok = pd.Series(True, index=df.index)
    if not market_cfg.get("enabled", False):
        return ok

    min_positive_ratio = market_cfg.get("min_positive_ratio_ma5")
    if min_positive_ratio is not None and "market_positive_ratio_ma5" in df.columns:
        ok &= df["market_positive_ratio_ma5"].fillna(1.0) >= float(min_positive_ratio)

    if market_cfg.get("avoid_weak_breadth", True) and "market_breadth_regime" in df.columns:
        ok &= df["market_breadth_regime"].fillna("") != "weak_breadth"

    if (
        market_cfg.get("avoid_range_neutral", True)
        and "market_trend_regime" in df.columns
        and "market_liquidity_regime" in df.columns
    ):
        ok &= ~(
            (df["market_trend_regime"].fillna("") == "range")
            & (df["market_liquidity_regime"].fillna("") == "neutral")
        )

    return ok.fillna(False)


def build_market_size_mult(df: pd.DataFrame, cfg: dict) -> pd.Series:
    sizing_cfg = cfg.get("market_sizing", {})
    mult = pd.Series(1.0, index=df.index)
    if not sizing_cfg.get("enabled", False):
        return mult

    breadth_map = {
        "strong_breadth": float(sizing_cfg.get("strong_breadth_mult", 1.0)),
        "neutral_breadth": float(sizing_cfg.get("neutral_breadth_mult", 0.80)),
        "weak_breadth": float(sizing_cfg.get("weak_breadth_mult", 0.50)),
    }
    trend_map = {
        "uptrend": float(sizing_cfg.get("uptrend_mult", 1.0)),
        "range": float(sizing_cfg.get("range_mult", 0.80)),
        "downtrend": float(sizing_cfg.get("downtrend_mult", 0.60)),
    }
    liquidity_map = {
        "expanding": float(sizing_cfg.get("expanding_mult", 1.0)),
        "neutral": float(sizing_cfg.get("neutral_liquidity_mult", 0.90)),
        "shrinking": float(sizing_cfg.get("shrinking_mult", 0.80)),
    }

    if "market_breadth_regime" in df.columns:
        mult *= df["market_breadth_regime"].map(breadth_map).fillna(1.0)
    if "market_trend_regime" in df.columns:
        mult *= df["market_trend_regime"].map(trend_map).fillna(1.0)
    if "market_liquidity_regime" in df.columns:
        mult *= df["market_liquidity_regime"].map(liquidity_map).fillna(1.0)

    range_neutral_mult = sizing_cfg.get("range_neutral_mult")
    if (
        range_neutral_mult is not None
        and "market_trend_regime" in df.columns
        and "market_liquidity_regime" in df.columns
    ):
        range_neutral = (df["market_trend_regime"] == "range") & (df["market_liquidity_regime"] == "neutral")
        mult = mult.where(~range_neutral, np.minimum(mult, float(range_neutral_mult)))

    return mult.clip(
        lower=float(sizing_cfg.get("min_mult", 0.40)),
        upper=float(sizing_cfg.get("max_mult", 1.0)),
    ).fillna(1.0)


def select_buy_rows(day: pd.DataFrame, held_codes: set[str], sell_codes: set[str], slots: int) -> pd.DataFrame:
    if slots <= 0:
        return day.iloc[0:0]
    rows = day[day["buy_signal"]].copy()
    if rows.empty:
        return rows
    unavailable = held_codes.difference(sell_codes)
    rows = rows[~rows["code"].isin(unavailable)]
    rows = rows.sort_values(["score", "amount"], ascending=[False, False])
    return rows.head(slots)
