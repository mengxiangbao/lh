from copy import deepcopy

import pandas as pd

from dragon_backtest.backtester import run_backtest
from dragon_backtest.config import DEFAULT_CONFIG


def _cfg(max_positions: int = 1) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["trade"]["initial_cash"] = 200000.0
    cfg["trade"]["max_positions"] = max_positions
    cfg["trade"]["target_weight"] = 1.0
    cfg["trade"]["buy_slippage"] = 0.0
    cfg["trade"]["sell_slippage"] = 0.0
    cfg["trade"]["commission_rate"] = 0.0
    cfg["trade"]["transfer_fee_rate"] = 0.0
    cfg["trade"]["min_commission"] = 0.0
    cfg["trade"]["impact_k"] = 0.0
    cfg["trade"]["volume_cap"] = 1.0
    cfg["trade"]["capacity_base"] = "trade"
    cfg["risk"]["max_holding_days"] = 999
    cfg["risk"]["stale_exit_days"] = 999
    cfg["risk"]["stop_loss"] = -0.99
    cfg["risk"]["trailing_stop"] = -0.99
    return cfg


def _signals_single_code(code: str = "000001", ts_code: str = "000001.SZ") -> pd.DataFrame:
    rows = [
        {
            "date": pd.Timestamp("2025-01-02"),
            "code": code,
            "ts_code": ts_code,
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "pre_close": 9.9,
            "up_limit": 11.0,
            "down_limit": 9.0,
            "amount": 1000000.0,
            "volume": 100000.0,
            "paused": False,
            "buy_signal": True,
            "trigger": True,
            "score": 0.9,
            "target_weight_mult": 1.0,
            "market_size_mult": 1.0,
            "candidate": True,
            "pre_pool": True,
            "name": "AAA",
            "exchange": "SZSE",
            "market": "主板",
            "sector": "X",
        },
        {
            "date": pd.Timestamp("2025-01-03"),
            "code": code,
            "ts_code": ts_code,
            "open": 10.9,
            "high": 11.4,
            "low": 10.8,
            "close": 11.1,
            "pre_close": 10.0,
            "up_limit": 11.0,
            "down_limit": 9.0,
            "amount": 1000000.0,
            "volume": 100000.0,
            "paused": False,
            "buy_signal": False,
            "trigger": False,
            "score": 0.1,
            "target_weight_mult": 1.0,
            "market_size_mult": 1.0,
            "candidate": False,
            "pre_pool": False,
            "name": "AAA",
            "exchange": "SZSE",
            "market": "主板",
            "sector": "X",
        },
    ]
    return pd.DataFrame(rows)


def _write_minute_csv(
    tmp_path,
    ts_code: str = "000001.SZ",
    freq: str = "5min",
    rows: list[dict] | None = None,
) -> None:
    minute_dir = tmp_path / freq
    minute_dir.mkdir(parents=True, exist_ok=True)
    rows = rows or []
    pd.DataFrame(rows).to_csv(
        minute_dir / f"{ts_code}_{freq}_20250103_090000_20250103_190000.csv",
        index=False,
    )


def test_minute_confirmed_error_when_missing_minute_data(tmp_path):
    cfg = _cfg()
    signals = _signals_single_code()
    result = run_backtest(
        signals=signals,
        cfg=cfg,
        mode="confirmed",
        execution_mode="minute_confirmed",
        minute_cfg={
            "minute_dir": str(tmp_path),
            "minute_freq": "5min",
            "minute_entry_time": "09:35",
            "missing_policy": "error",
            "price_field": "open",
        },
    )
    trades = result["trades"]
    assert len(trades) == 1
    assert trades.iloc[0]["status"] == "blocked"
    assert trades.iloc[0]["blocked_reason"] == "missing_minute_data"


def test_minute_confirmed_fallback_to_daily_when_missing_minute_data(tmp_path):
    cfg = _cfg()
    signals = _signals_single_code()
    result = run_backtest(
        signals=signals,
        cfg=cfg,
        mode="confirmed",
        execution_mode="minute_confirmed",
        minute_cfg={
            "minute_dir": str(tmp_path),
            "minute_freq": "5min",
            "minute_entry_time": "09:35",
            "missing_policy": "fallback_daily",
            "price_field": "open",
        },
    )
    filled = result["trades"][result["trades"]["status"] == "filled"]
    assert len(filled) == 1
    assert filled.iloc[0]["execution_mode"] == "daily_fallback"
    assert filled.iloc[0]["minute_fallback"] == "missing_minute_data"


def test_minute_confirmed_uses_first_liquid_bar_after_cutoff(tmp_path):
    cfg = _cfg()
    signals = _signals_single_code()
    _write_minute_csv(
        tmp_path=tmp_path,
        rows=[
            {"datetime": "2025-01-03 09:30:00", "open": 10.3, "close": 10.4, "amount": 120000.0, "volume": 10000.0},
            {"datetime": "2025-01-03 09:35:00", "open": 10.5, "close": 10.55, "amount": 130000.0, "volume": 10000.0},
            {"datetime": "2025-01-03 09:40:00", "open": 10.6, "close": 10.65, "amount": 140000.0, "volume": 10000.0},
        ],
    )
    result = run_backtest(
        signals=signals,
        cfg=cfg,
        mode="confirmed",
        execution_mode="minute_confirmed",
        minute_cfg={
            "minute_dir": str(tmp_path),
            "minute_freq": "5min",
            "minute_entry_time": "09:35",
            "missing_policy": "error",
            "price_field": "open",
        },
    )
    filled = result["trades"][result["trades"]["status"] == "filled"]
    assert len(filled) == 1
    assert abs(float(filled.iloc[0]["price"]) - 10.5) < 1e-9
    assert str(pd.Timestamp(filled.iloc[0]["minute_bar_time"])) == "2025-01-03 09:35:00"
    assert filled.iloc[0]["execution_mode"] == "minute_confirmed"


def test_minute_confirmed_blocks_when_no_liquidity_after_cutoff(tmp_path):
    cfg = _cfg()
    signals = _signals_single_code()
    _write_minute_csv(
        tmp_path=tmp_path,
        rows=[
            {"datetime": "2025-01-03 09:35:00", "open": 10.5, "close": 10.55, "amount": 0.0, "volume": 10000.0},
            {"datetime": "2025-01-03 09:40:00", "open": 10.6, "close": 10.65, "amount": 150000.0, "volume": 0.0},
        ],
    )
    result = run_backtest(
        signals=signals,
        cfg=cfg,
        mode="confirmed",
        execution_mode="minute_confirmed",
        minute_cfg={
            "minute_dir": str(tmp_path),
            "minute_freq": "5min",
            "minute_entry_time": "09:35",
            "missing_policy": "error",
            "price_field": "open",
        },
    )
    trades = result["trades"]
    assert len(trades) == 1
    assert trades.iloc[0]["status"] == "blocked"
    assert trades.iloc[0]["blocked_reason"] == "minute_no_liquidity"
