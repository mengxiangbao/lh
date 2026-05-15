from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import write_table


SECTORS = [
    "robotics",
    "ai_compute",
    "semiconductor",
    "new_energy",
    "medicine",
    "military",
]


def generate_sample_daily(
    out_dir: str | Path = "data/raw",
    stocks: int = 96,
    days: int = 520,
    seed: int = 7,
) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2021-01-04", periods=days)
    codes = [f"{i:06d}" for i in range(1, stocks + 1)]
    sector_map = {code: SECTORS[i % len(SECTORS)] for i, code in enumerate(codes)}

    sector_ret = {}
    for sector in SECTORS:
        base = rng.normal(0.0001, 0.010, days)
        waves = 0.004 * np.sin(np.linspace(0, 10 * np.pi, days) + rng.uniform(0, 2))
        sector_ret[sector] = base + waves

    rows = []
    dragon_codes = set(codes[::17])
    start_points = {code: rng.integers(180, days - 80) for code in dragon_codes}

    for idx, code in enumerate(codes):
        sector = sector_map[code]
        price = rng.uniform(8, 35)
        float_mv = rng.uniform(35e8, 180e8)
        total_mv = float_mv * rng.uniform(1.1, 1.8)
        list_date = dates[0] - pd.Timedelta(days=int(rng.integers(300, 1800)))
        stock_noise = rng.normal(0.0, 0.018, days)
        beta = rng.uniform(0.7, 1.3)
        start_idx = start_points.get(code)

        for i, date in enumerate(dates):
            pre_close = price
            daily_ret = beta * sector_ret[sector][i] + stock_noise[i]

            if start_idx is not None:
                if start_idx - 25 <= i < start_idx:
                    daily_ret += 0.002 + max(sector_ret[sector][i], 0) * 0.5
                elif i == start_idx:
                    daily_ret += 0.070
                elif start_idx < i <= start_idx + 22:
                    daily_ret += 0.011 + rng.normal(0, 0.006)
                elif start_idx + 22 < i <= start_idx + 35:
                    daily_ret += rng.normal(-0.002, 0.014)

            daily_ret = float(np.clip(daily_ret, -0.095, 0.095))
            open_ret = float(np.clip(rng.normal(daily_ret * 0.25, 0.006), -0.08, 0.08))
            open_price = pre_close * (1 + open_ret)
            close = pre_close * (1 + daily_ret)
            intraday_span = abs(daily_ret) + rng.uniform(0.006, 0.035)
            high = max(open_price, close) * (1 + intraday_span * rng.uniform(0.2, 0.8))
            low = min(open_price, close) * (1 - intraday_span * rng.uniform(0.2, 0.8))
            up_limit = round(pre_close * 1.10, 2)
            down_limit = round(pre_close * 0.90, 2)
            high = min(high, up_limit)
            low = max(low, down_limit)
            open_price = min(max(open_price, down_limit), up_limit)
            close = min(max(close, down_limit), up_limit)

            base_amount = rng.lognormal(np.log(1.8e8), 0.45)
            if start_idx is not None:
                if start_idx - 20 <= i < start_idx:
                    base_amount *= 1.25
                elif i == start_idx:
                    base_amount *= 2.8
                elif start_idx < i <= start_idx + 15:
                    base_amount *= 2.0
            amount = float(base_amount * (1 + abs(daily_ret) * 5))
            volume = amount / max(close, 0.01)
            turnover = amount / float_mv

            paused = bool(rng.random() < 0.001)
            is_st = bool(rng.random() < 0.002)
            if paused:
                open_price = high = low = close = pre_close
                amount = 0.0
                volume = 0.0

            rows.append(
                {
                    "date": date,
                    "code": code,
                    "ts_code": f"{code}.SH" if code.startswith("6") else f"{code}.SZ",
                    "name": f"样例{code}",
                    "exchange": "SSE" if code.startswith("6") else "SZSE",
                    "market": "sample",
                    "open": round(open_price, 3),
                    "high": round(high, 3),
                    "low": round(low, 3),
                    "close": round(close, 3),
                    "pre_close": round(pre_close, 3),
                    "volume": round(volume, 0),
                    "amount": round(amount, 2),
                    "up_limit": up_limit,
                    "down_limit": down_limit,
                    "paused": paused,
                    "is_st": is_st,
                    "list_date": list_date,
                    "sector": sector,
                    "float_mv": round(float_mv, 2),
                    "total_mv": round(total_mv, 2),
                    "turnover": turnover,
                }
            )
            price = close

    df = pd.DataFrame(rows)
    path = out_dir / "daily_price.csv"
    write_table(df, path)
    return path
