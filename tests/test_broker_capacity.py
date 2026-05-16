import pandas as pd

from dragon_backtest.broker import execute_buy


def _base_row(amount: float = 100000.0) -> pd.Series:
    return pd.Series(
        {
            "date": pd.Timestamp("2024-01-02"),
            "code": "000001",
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.1,
            "pre_close": 10.0,
            "up_limit": 11.0,
            "down_limit": 9.0,
            "volume": 100000.0,
            "amount": amount,
            "paused": False,
        }
    )


def _base_signal_row(amount: float = 200000.0) -> pd.Series:
    return pd.Series(
        {
            "date": pd.Timestamp("2024-01-01"),
            "code": "000001",
            "score": 0.9,
            "amount": amount,
            "trigger": True,
            "target_weight_mult": 1.0,
            "market_size_mult": 1.0,
        }
    )


def _trade_cfg(capacity_base: str) -> dict:
    return {
        "capacity_base": capacity_base,
        "volume_cap": 0.1,
        "buy_slippage": 0.0,
        "sell_slippage": 0.0,
        "commission_rate": 0.0,
        "min_commission": 0.0,
        "transfer_fee_rate": 0.0,
        "stamp_tax_rate_before_20230828": 0.001,
        "stamp_tax_rate_after_20230828": 0.0005,
        "impact_k": 0.0,
        "lot_size": 100,
        "limit_tolerance": 0.0001,
    }


def test_capacity_base_signal_uses_signal_amount():
    row = _base_row(amount=100000.0)
    signal_row = _base_signal_row(amount=200000.0)
    position, _, record = execute_buy(
        row=row,
        signal_row=signal_row,
        trade_date=pd.Timestamp("2024-01-02"),
        trade_index=1,
        target_value=10000.0,
        cash=10000.0,
        trade_cfg=_trade_cfg("signal"),
    )
    assert position is not None
    assert record["capacity_base"] == 200000.0
    assert record["value"] == 10000.0


def test_capacity_base_trade_uses_trade_amount():
    row = _base_row(amount=100000.0)
    signal_row = _base_signal_row(amount=200000.0)
    position, _, record = execute_buy(
        row=row,
        signal_row=signal_row,
        trade_date=pd.Timestamp("2024-01-02"),
        trade_index=1,
        target_value=10000.0,
        cash=10000.0,
        trade_cfg=_trade_cfg("trade"),
    )
    assert position is not None
    assert record["capacity_base"] == 100000.0
    assert record["value"] == 10000.0


def test_capacity_base_min_signal_trade_uses_min():
    row = _base_row(amount=100000.0)
    signal_row = _base_signal_row(amount=200000.0)
    position, _, record = execute_buy(
        row=row,
        signal_row=signal_row,
        trade_date=pd.Timestamp("2024-01-02"),
        trade_index=1,
        target_value=10000.0,
        cash=10000.0,
        trade_cfg=_trade_cfg("min_signal_trade"),
    )
    assert position is not None
    assert record["capacity_base"] == 100000.0
    assert record["value"] == 10000.0
