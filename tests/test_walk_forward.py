import pandas as pd

from dragon_backtest.parameter_sweep import build_walk_forward_windows


def test_build_walk_forward_windows_basic():
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    df = pd.DataFrame({"date": dates, "code": "000001"})
    windows = build_walk_forward_windows(df, train_months=12, test_months=3)
    assert len(windows) > 0
    train_start, train_end, test_start, test_end = windows[0]
    assert train_start < train_end < test_start <= test_end
