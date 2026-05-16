import pandas as pd

from dragon_backtest.data_check import check_daily_data


def _base_rows() -> list[dict]:
    return [
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


def _run_check(tmp_path, rows):
    path = tmp_path / "daily.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return check_daily_data(path)


def test_duplicate_date_code_is_error(tmp_path):
    rows = _base_rows() * 2
    report = _run_check(tmp_path, rows)
    checks = {i["check"]: i for i in report["issues"]}
    assert "date_code.duplicate" in checks
    assert checks["date_code.duplicate"]["severity"] == "error"


def test_amount_unit_suspicious_small_warns(tmp_path):
    rows = _base_rows()
    rows[0]["amount"] = 5000.0
    report = _run_check(tmp_path, rows)
    checks = {i["check"]: i for i in report["issues"]}
    assert "amount.unit_suspicious_small" in checks
    assert checks["amount.unit_suspicious_small"]["severity"] == "warning"


def test_sector_too_few_daily_groups_warns(tmp_path):
    rows = []
    for d in ["2024-01-02", "2024-01-03", "2024-01-04"]:
        for code in ["000001", "000002", "000003"]:
            row = _base_rows()[0].copy()
            row["date"] = d
            row["code"] = code
            row["sector"] = "OnlyOneSector"
            rows.append(row)
    report = _run_check(tmp_path, rows)
    checks = {i["check"]: i for i in report["issues"]}
    assert "sector.too_few_daily_groups" in checks
    assert checks["sector.too_few_daily_groups"]["severity"] == "warning"
