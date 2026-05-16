import pandas as pd

from dragon_backtest.data import compute_data_hash
from dragon_backtest.features import prepare_features_cached


def _daily_df() -> pd.DataFrame:
    rows = []
    for i in range(80):
        d = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        rows.append(
            {
                "date": d,
                "code": "000001",
                "open": 10.0 + i * 0.01,
                "high": 10.2 + i * 0.01,
                "low": 9.8 + i * 0.01,
                "close": 10.1 + i * 0.01,
                "pre_close": 10.0 + i * 0.01,
                "volume": 100000 + i * 10,
                "amount": 2e8 + i * 10000,
                "up_limit": 11.0 + i * 0.01,
                "down_limit": 9.0 + i * 0.01,
                "paused": False,
                "is_st": False,
                "list_date": pd.Timestamp("2020-01-01"),
                "sector": "Bank",
                "float_mv": 1e10,
                "total_mv": 1.2e10,
                "turnover": 0.01,
                "ts_code": "000001.SZ",
                "name": "A",
                "exchange": "SZSE",
                "market": "主板",
            }
        )
    return pd.DataFrame(rows)


def test_prepare_features_cached_creates_and_hits_cache(tmp_path):
    daily = _daily_df()
    data_hash = compute_data_hash(df=daily)
    cache_dir = tmp_path / "feature_store"

    features_1, cache_path_1 = prepare_features_cached(
        daily=daily,
        cache_dir=str(cache_dir),
        data_hash=data_hash,
        float32=False,
    )
    assert cache_path_1 is not None
    assert (cache_dir / f"daily_hash={data_hash}" / "features.parquet").exists()

    features_2, cache_path_2 = prepare_features_cached(
        daily=daily,
        cache_dir=str(cache_dir),
        data_hash=data_hash,
        float32=False,
    )
    assert cache_path_2 is not None
    assert len(features_1) == len(features_2)
