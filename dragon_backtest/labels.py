from __future__ import annotations

import numpy as np
import pandas as pd


def _future_rolling_max(s: pd.Series, window: int) -> pd.Series:
    future = s.shift(-1)
    return future.iloc[::-1].rolling(window, min_periods=1).max().iloc[::-1]


def _future_compound_return(ret: pd.Series, window: int) -> pd.Series:
    future = (1 + ret.shift(-1)).iloc[::-1]
    prod = future.rolling(window, min_periods=1).apply(np.prod, raw=True).iloc[::-1]
    return prod - 1


def _days_to_next_start(start: pd.Series) -> pd.Series:
    values = start.astype(bool).to_numpy()
    n = len(values)
    out = np.full(n, np.nan)
    next_idx = np.nan
    for i in range(n - 1, -1, -1):
        out[i] = next_idx - i if not np.isnan(next_idx) else np.nan
        if values[i]:
            next_idx = i
    return pd.Series(out, index=start.index)


def build_research_labels(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy().sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code", group_keys=False)

    df["future_max_close_20d"] = g["close"].transform(lambda s: _future_rolling_max(s, 20))
    df["future_max_return_20d"] = df["future_max_close_20d"] / df["close"].replace(0, np.nan) - 1
    df["future_return_20d"] = g["ret_1d"].transform(lambda s: _future_compound_return(s, 20))
    df["future_rank_in_sector_20d"] = df.groupby(["date", "sector"])["future_return_20d"].rank(
        pct=True, ascending=False
    )

    sector = df[["date", "sector", "sector_ret_1d"]].drop_duplicates().sort_values(["sector", "date"])
    sector["future_sector_return_20d"] = sector.groupby("sector", group_keys=False)["sector_ret_1d"].transform(
        lambda s: _future_compound_return(s, 20)
    )
    df = df.merge(sector[["date", "sector", "future_sector_return_20d"]], on=["date", "sector"], how="left")
    df["future_excess_return_20d"] = df["future_return_20d"] - df["future_sector_return_20d"]

    df["research_start_day"] = (
        (df["close"] > df["high_60d_prev"])
        & (df["amount_ma5_to_ma60"] > 1.5)
        & ((df["ret_1d"] > 0.05) | df["is_limit_up"].astype(bool))
        & (df["sector_shift_pct"] >= 0.70)
        & (df["future_max_return_20d"] > 0.25)
        & (df["future_rank_in_sector_20d"] <= 0.10)
    ).fillna(False)

    df["days_to_start"] = df.groupby("code", group_keys=False)["research_start_day"].transform(_days_to_next_start)
    df["y_start_5d"] = ((df["days_to_start"] > 0) & (df["days_to_start"] <= 5)).astype(int)
    df["y_start_10d"] = ((df["days_to_start"] > 0) & (df["days_to_start"] <= 10)).astype(int)
    df["y_leader_20d"] = ((df["future_max_return_20d"] > 0.25) & (df["future_rank_in_sector_20d"] <= 0.10)).astype(int)

    df["label_class"] = "gray"
    positive = (df["y_start_10d"] == 1) & (df["y_leader_20d"] == 1)
    negative = (df["future_max_return_20d"] < 0.10) & (df["future_excess_return_20d"] <= 0)
    df.loc[positive, "label_class"] = "positive"
    df.loc[negative, "label_class"] = "negative"
    df["sample_weight"] = np.where(df["label_class"].eq("gray"), 0.25, 1.0)
    return df.sort_values(["date", "code"]).reset_index(drop=True)

