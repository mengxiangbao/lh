import pandas as pd

from dragon_backtest.data_check import check_daily_data


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "code": "000001",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "pre_close": 10.0,
                "volume": 100000.0,
                "amount": 2e8,
                "up_limit": 11.0,
                "down_limit": 9.0,
                "paused": 0,
                "is_st": 0,
                "list_date": "2020-01-01",
                "sector": "Bank",
                "float_mv": 1e10,
                "total_mv": 1.2e10,
                "turnover": 0.01,
            }
        ]
    )


def test_point_in_time_checks_group_present(tmp_path):
    df = _base_df()
    path = tmp_path / "daily.csv"
    df.to_csv(path, index=False)
    report = check_daily_data(path)
    assert "point_in_time_checks" in report
    checks = [x["check"] for x in report["point_in_time_checks"]]
    assert "point_in_time.sector_effective_date.missing" in checks
    assert "point_in_time.trade_status.missing" in checks


def test_adjustment_limit_mismatch_raises_error(tmp_path):
    df = _base_df()
    df["adjustment_mode"] = "qfq"
    df["limit_price_mode"] = "raw"
    df["sector_source"] = "provider_x"
    df["sector_effective_date"] = "2024-01-01"
    df["trade_status"] = "L"
    df["delist_date"] = ""
    path = tmp_path / "daily.csv"
    df.to_csv(path, index=False)
    report = check_daily_data(path)
    checks = {x["check"]: x for x in report["issues"]}
    assert "point_in_time.limit_adjustment_mismatch" in checks
    assert checks["point_in_time.limit_adjustment_mismatch"]["severity"] == "error"
