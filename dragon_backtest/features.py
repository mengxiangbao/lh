from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(a, b):
    return a / pd.Series(b).replace(0, np.nan)


def _is_limit_up_by_market_fallback(df: pd.DataFrame) -> pd.Series:
    market_text = df.get("market", pd.Series("", index=df.index)).fillna("").astype(str)
    exchange_text = df.get("exchange", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    is_st = df.get("is_st", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    list_days = pd.to_numeric(df.get("list_days", pd.Series(np.nan, index=df.index)), errors="coerce")
    ret_1d = pd.to_numeric(df.get("ret_1d", pd.Series(np.nan, index=df.index)), errors="coerce")

    market_kcb = market_text.str.contains("科创板", regex=False)
    market_gem = market_text.str.contains("创业板", regex=False)
    market_bse = market_text.str.contains("北交所", regex=False) | (exchange_text == "BSE")
    market_main = market_text.str.contains("主板", regex=False) | market_text.str.contains("中小板", regex=False) | market_text.eq("")
    new_listing = list_days.notna() & (list_days < 5)

    threshold = pd.Series(0.10, index=df.index, dtype=float)
    threshold[is_st] = 0.05
    threshold[market_kcb | market_gem] = 0.20
    threshold[market_bse] = 0.30
    threshold[new_listing & market_main] = np.nan
    return ret_1d >= (threshold - 0.001)


def compute_is_limit_up(df: pd.DataFrame) -> pd.Series:
    up_limit = pd.to_numeric(df.get("up_limit", pd.Series(np.nan, index=df.index)), errors="coerce")
    pre_close = pd.to_numeric(df.get("pre_close", pd.Series(np.nan, index=df.index)), errors="coerce")
    close = pd.to_numeric(df.get("close", pd.Series(np.nan, index=df.index)), errors="coerce")
    valid_limit = (
        up_limit.notna()
        & pre_close.notna()
        & close.notna()
        & (up_limit > 0)
        & (up_limit < 9999)
        & (pre_close > 0)
        & ((up_limit / pre_close - 1) < 5)
    )
    by_limit = close >= up_limit * 0.999
    by_fallback = _is_limit_up_by_market_fallback(df)
    return by_limit.where(valid_limit, by_fallback).fillna(False)


def compound_return(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(2, window // 2)
    return (1 + s).rolling(window, min_periods=min_periods).apply(np.prod, raw=True) - 1


def rolling_sum_masked(values: pd.Series, mask: pd.Series, window: int, min_periods: int = 3) -> pd.Series:
    return values.where(mask, 0.0).rolling(window, min_periods=min_periods).sum()


def rank_by_date(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    return df.groupby("date")[col].rank(pct=True, ascending=ascending)


def prepare_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code", group_keys=False)

    df["list_days"] = (df["date"] - df["list_date"]).dt.days
    df["ret_1d"] = df["close"] / df["pre_close"].replace(0, np.nan) - 1
    df["ret_5d"] = g["ret_1d"].transform(lambda s: compound_return(s, 5, 3))
    df["ret_10d"] = g["ret_1d"].transform(lambda s: compound_return(s, 10, 5))
    df["ret_20d"] = g["ret_1d"].transform(lambda s: compound_return(s, 20, 10))

    for window in [5, 10, 20, 60, 120, 250]:
        df[f"ma{window}"] = g["close"].transform(lambda s, w=window: s.rolling(w, min_periods=max(2, w // 3)).mean())
    for window in [5, 20, 60]:
        df[f"amount_ma{window}"] = g["amount"].transform(lambda s, w=window: s.rolling(w, min_periods=max(2, w // 3)).mean())

    df["high_20d"] = g["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    df["low_20d"] = g["low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    df["high_60d_prev"] = g["close"].transform(lambda s: s.rolling(60, min_periods=20).max().shift(1))
    df["high_120d"] = g["close"].transform(lambda s: s.rolling(120, min_periods=40).max())
    df["high_250d"] = g["close"].transform(lambda s: s.rolling(250, min_periods=80).max())
    df["close_to_high_60d"] = df["close"] / df["high_60d_prev"].replace(0, np.nan)
    df["close_position_20d"] = (df["close"] - df["low_20d"]) / (df["high_20d"] - df["low_20d"]).replace(0, np.nan)
    df["close_position_20d"] = df["close_position_20d"].clip(0, 1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["pre_close"]).abs()
    tr3 = (df["low"] - df["pre_close"]).abs()
    df["true_range"] = np.maximum.reduce([tr1.to_numpy(), tr2.to_numpy(), tr3.to_numpy()])
    df["atr20"] = g["true_range"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["atr20_to_close"] = df["atr20"] / df["close"].replace(0, np.nan)
    df["ret_vol_20d"] = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=10).std())
    df["range_20d"] = df["high_20d"] / df["low_20d"].replace(0, np.nan) - 1
    df["boll_width_20d"] = 4 * df["ret_vol_20d"]

    rolling_max = g["close"].transform(lambda s: s.rolling(20, min_periods=10).max())
    drawdown = df["close"] / rolling_max.replace(0, np.nan) - 1
    df["max_drawdown_20d"] = drawdown.groupby(df["code"], group_keys=False).transform(
        lambda s: s.rolling(20, min_periods=10).min()
    )

    df["is_limit_up"] = compute_is_limit_up(df)
    df["touch_limit_up"] = df["high"] >= df["up_limit"] * 0.999
    df["limit_up_count_10d"] = g["is_limit_up"].transform(lambda s: s.rolling(10, min_periods=1).sum())
    df["limit_up_count_250d"] = g["is_limit_up"].transform(lambda s: s.rolling(250, min_periods=30).sum())
    df["touch_limit_count_250d"] = g["touch_limit_up"].transform(lambda s: s.rolling(250, min_periods=30).sum())

    up_amount = df["amount"].where(df["ret_1d"] > 0, 0.0)
    down_amount = df["amount"].where(df["ret_1d"] <= 0, 0.0)
    df["up_amount_sum_20d"] = up_amount.groupby(df["code"], group_keys=False).transform(lambda s: s.rolling(20, min_periods=10).sum())
    df["down_amount_sum_20d"] = down_amount.groupby(df["code"], group_keys=False).transform(lambda s: s.rolling(20, min_periods=10).sum())
    df["up_amount_ratio_20d"] = df["up_amount_sum_20d"] / df["down_amount_sum_20d"].replace(0, np.nan)

    df["obv_step"] = np.sign(df["ret_1d"].fillna(0)) * df["volume"]
    df["obv"] = df.groupby("code")["obv_step"].cumsum()
    obv_shift = g["obv"].shift(20)
    volume_sum_20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).sum())
    df["obv_slope_20d"] = (df["obv"] - obv_shift) / volume_sum_20.replace(0, np.nan)

    df["close_location_value"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    ).replace(0, np.nan)
    df["close_location_value"] = df["close_location_value"].clip(-1, 1)
    mfv = df["close_location_value"].fillna(0) * df["volume"]
    df["cmf_20d"] = mfv.groupby(df["code"], group_keys=False).transform(lambda s: s.rolling(20, min_periods=10).sum()) / volume_sum_20.replace(0, np.nan)

    df = add_market_features(df)
    df = add_sector_features(df)
    df = add_factor_scores(df)
    return df.sort_values(["date", "code"]).reset_index(drop=True)


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["ret_x_amount"] = working["ret_1d"].fillna(0.0) * working["amount"].fillna(0.0)
    market = (
        working.groupby("date", as_index=False)
        .agg(
            market_ret_1d=("ret_1d", "mean"),
            market_amount=("amount", "sum"),
            ret_x_amount=("ret_x_amount", "sum"),
            market_positive_ratio=("ret_1d", lambda s: float((s > 0).mean())),
            market_strong_ratio=("ret_1d", lambda s: float((s > 0.03).mean())),
            market_weak_ratio=("ret_1d", lambda s: float((s < -0.03).mean())),
            market_limit_up_count=("is_limit_up", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    market["market_ret_amount_weighted"] = market["ret_x_amount"] / market["market_amount"].replace(0, np.nan)
    market["market_ret_5d"] = compound_return(market["market_ret_1d"], 5, 3)
    market["market_ret_20d"] = compound_return(market["market_ret_1d"], 20, 10)
    market["market_amount_ma20"] = market["market_amount"].rolling(20, min_periods=10).mean()
    market["market_amount_ma60"] = market["market_amount"].rolling(60, min_periods=20).mean()
    market["market_amount_ratio"] = market["market_amount_ma20"] / market["market_amount_ma60"].replace(0, np.nan)
    market["market_positive_ratio_ma5"] = market["market_positive_ratio"].rolling(5, min_periods=3).mean()

    market["market_trend_regime"] = np.select(
        [
            market["market_ret_20d"] >= 0.05,
            market["market_ret_20d"] <= -0.05,
        ],
        ["uptrend", "downtrend"],
        default="range",
    )
    market["market_liquidity_regime"] = np.select(
        [
            market["market_amount_ratio"] >= 1.15,
            market["market_amount_ratio"] <= 0.85,
        ],
        ["expanding", "shrinking"],
        default="neutral",
    )
    market["market_breadth_regime"] = np.select(
        [
            market["market_positive_ratio_ma5"] >= 0.55,
            market["market_positive_ratio_ma5"] <= 0.45,
        ],
        ["strong_breadth", "weak_breadth"],
        default="neutral_breadth",
    )
    market["market_combined_regime"] = market["market_trend_regime"] + "|" + market["market_liquidity_regime"]

    keep = [
        "date",
        "market_ret_1d",
        "market_ret_5d",
        "market_ret_20d",
        "market_ret_amount_weighted",
        "market_amount",
        "market_amount_ratio",
        "market_positive_ratio",
        "market_positive_ratio_ma5",
        "market_strong_ratio",
        "market_weak_ratio",
        "market_limit_up_count",
        "market_trend_regime",
        "market_liquidity_regime",
        "market_breadth_regime",
        "market_combined_regime",
    ]
    return working.drop(columns=["ret_x_amount"]).merge(market[keep], on="date", how="left")


def add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["above_ma20"] = working["close"] > working["ma20"]
    working["is_new_high20"] = working["close"] >= working["high_20d"] * 0.999
    sector = (
        working.groupby(["date", "sector"], as_index=False)
        .agg(
            sector_ret_1d=("ret_1d", "mean"),
            sector_amount=("amount", "sum"),
            pct_above_ma20=("above_ma20", "mean"),
            new_high_20d_count=("is_new_high20", "sum"),
            sector_limit_up_count=("is_limit_up", "sum"),
            sector_max_drawdown_20d_median=("max_drawdown_20d", "median"),
        )
        .sort_values(["sector", "date"])
    )
    sg = sector.groupby("sector", group_keys=False)
    sector["sector_ret_5d"] = sg["sector_ret_1d"].transform(lambda s: compound_return(s, 5, 3))
    sector["sector_ret_20d"] = sg["sector_ret_1d"].transform(lambda s: compound_return(s, 20, 10))
    sector["sector_amount_ma5"] = sg["sector_amount"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    sector["sector_amount_ma60"] = sg["sector_amount"].transform(lambda s: s.rolling(60, min_periods=20).mean())
    sector["sector_amount_ma5_to_ma60"] = sector["sector_amount_ma5"] / sector["sector_amount_ma60"].replace(0, np.nan)
    sector["sector_limit_up_count_5d"] = sg["sector_limit_up_count"].transform(lambda s: s.rolling(5, min_periods=2).sum())
    sector["sector_limit_up_count_5d_change"] = sector["sector_limit_up_count_5d"] - sg["sector_limit_up_count_5d"].shift(5)
    sector["sector_ret_20d_pct"] = sector.groupby("date")["sector_ret_20d"].rank(pct=True, ascending=True)
    sector["sector_ret_1d_pct"] = sector.groupby("date")["sector_ret_1d"].rank(pct=True, ascending=True)
    sector["sector_ret_20d_pct_change"] = sector["sector_ret_20d_pct"] - sg["sector_ret_20d_pct"].shift(20)

    components = []
    for col in [
        "sector_ret_20d_pct_change",
        "sector_amount_ma5_to_ma60",
        "pct_above_ma20",
        "new_high_20d_count",
        "sector_limit_up_count_5d_change",
    ]:
        rank_col = f"{col}_rank"
        sector[rank_col] = sector.groupby("date")[col].rank(pct=True, ascending=True)
        components.append(rank_col)
    sector["sector_overheat_penalty"] = (sector["sector_ret_20d_pct"] > 0.95).astype(float) * 0.20
    sector["sector_shift_raw"] = sector[components].mean(axis=1) - sector["sector_overheat_penalty"]
    sector["sector_shift_pct"] = sector.groupby("date")["sector_shift_raw"].rank(pct=True, ascending=True)
    sector["sector_shift_rank"] = 1 - sector["sector_shift_pct"]

    keep = [
        "date",
        "sector",
        "sector_ret_1d",
        "sector_ret_5d",
        "sector_ret_20d",
        "sector_ret_1d_pct",
        "sector_ret_20d_pct",
        "sector_amount_ma5_to_ma60",
        "sector_shift_pct",
        "sector_shift_rank",
        "sector_max_drawdown_20d_median",
    ]
    merged = working.merge(sector[keep], on=["date", "sector"], how="left")
    return merged


def add_factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rel_ret_5d"] = df["ret_5d"] - df["sector_ret_5d"]
    df["rel_ret_20d"] = df["ret_20d"] - df["sector_ret_20d"]
    df["ret_20d_rank_in_sector"] = df.groupby(["date", "sector"])["ret_20d"].rank(pct=True, ascending=False)
    df["ret_20d_sector_score"] = 1 - df["ret_20d_rank_in_sector"].fillna(1)

    down_mask = df["sector_ret_1d"] < 0
    df["excess_ret_1d"] = df["ret_1d"] - df["sector_ret_1d"]
    df["down_excess_ret"] = df["excess_ret_1d"].where(down_mask)
    df["down_sector_ret"] = df["sector_ret_1d"].where(down_mask)
    g = df.groupby("code", group_keys=False)
    df["down_excess_mean_20d"] = g["down_excess_ret"].transform(lambda s: s.rolling(20, min_periods=3).mean())
    stock_down_sum = g["ret_1d"].transform(lambda s: s.where(down_mask.loc[s.index], 0.0).rolling(20, min_periods=3).sum())
    sector_down_sum = g["down_sector_ret"].transform(lambda s: s.fillna(0.0).rolling(20, min_periods=3).sum())
    df["downside_capture_20d"] = stock_down_sum / sector_down_sum.abs().replace(0, np.nan)

    df["amount_ma5_to_ma60"] = df["amount_ma5"] / df["amount_ma60"].replace(0, np.nan)

    rank_rel_5 = rank_by_date(df, "rel_ret_5d", True)
    rank_rel_20 = rank_by_date(df, "rel_ret_20d", True)
    rank_sector_pos = rank_by_date(df, "ret_20d_sector_score", True)
    hidden_penalty = (df["ret_20d"] > 0.30).astype(float) * 0.20
    df["hidden_rs_score"] = (rank_rel_5 + rank_rel_20 + rank_sector_pos) / 3 - hidden_penalty

    rank_down_excess = rank_by_date(df, "down_excess_mean_20d", True)
    rank_mdd_low = rank_by_date(df.assign(_neg_mdd=-df["max_drawdown_20d"].abs()), "_neg_mdd", True)
    rank_capture_low = rank_by_date(df.assign(_neg_capture=-df["downside_capture_20d"]), "_neg_capture", True)
    df["anti_fall_score"] = (rank_down_excess + rank_mdd_low + rank_capture_low) / 3

    rank_atr_low = rank_by_date(df.assign(_neg_atr=-df["atr20_to_close"]), "_neg_atr", True)
    rank_boll_low = rank_by_date(df.assign(_neg_boll=-df["boll_width_20d"]), "_neg_boll", True)
    rank_range_low = rank_by_date(df.assign(_neg_range=-df["range_20d"]), "_neg_range", True)
    rank_close_pos = rank_by_date(df, "close_position_20d", True)
    df["vol_squeeze_score"] = (rank_atr_low + rank_boll_low + rank_range_low + rank_close_pos) / 4

    rank_atr_high = rank_by_date(df, "atr20_to_close", True)
    rank_range_high = rank_by_date(df, "range_20d", True)
    rank_ret_10 = rank_by_date(df, "ret_10d", True)
    rank_limit_10 = rank_by_date(df, "limit_up_count_10d", True)
    volatility_overheat_penalty = (
        (df["ret_20d"] > 0.45).astype(float) * 0.20
        + (df["amount_ma5_to_ma60"] > 3).astype(float) * 0.20
    )
    df["volatility_energy_score"] = (
        0.30 * rank_atr_high
        + 0.30 * rank_range_high
        + 0.25 * rank_ret_10
        + 0.15 * rank_limit_10
        - volatility_overheat_penalty
    )

    rank_up_amount = rank_by_date(df, "up_amount_ratio_20d", True)
    rank_obv = rank_by_date(df, "obv_slope_20d", True)
    rank_cmf = rank_by_date(df, "cmf_20d", True)
    rank_clv = rank_by_date(df, "close_location_value", True)
    amount_overheat_penalty = (df["amount_ma5_to_ma60"] > 3).astype(float) * 0.20
    df["accumulation_score"] = (rank_up_amount + rank_obv + rank_cmf + rank_clv) / 4 - amount_overheat_penalty

    position_parts = [
        rank_by_date(df, "close_to_high_60d", True),
        rank_by_date(df.assign(_close_to_high120=df["close"] / df["high_120d"].replace(0, np.nan)), "_close_to_high120", True),
        rank_by_date(df.assign(_close_to_ma20=df["close"] / df["ma20"].replace(0, np.nan)), "_close_to_ma20", True),
        rank_by_date(df.assign(_ma20_to_ma60=df["ma20"] / df["ma60"].replace(0, np.nan)), "_ma20_to_ma60", True),
        rank_close_pos,
    ]
    df["position_score"] = sum(position_parts) / len(position_parts)

    rank_limit_count = rank_by_date(df, "limit_up_count_250d", True)
    rank_touch_count = rank_by_date(df, "touch_limit_count_250d", True)
    df["limit_gene_score"] = (rank_limit_count + rank_touch_count) / 2

    score_cols = [
        "hidden_rs_score",
        "anti_fall_score",
        "vol_squeeze_score",
        "volatility_energy_score",
        "accumulation_score",
        "position_score",
        "limit_gene_score",
        "sector_shift_pct",
    ]
    df[score_cols] = df[score_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1)
    return df
