import pandas as pd

from dragon_backtest.broker import Position, can_buy, can_sell, stamp_tax_rate, trade_cost


def _trade_cfg() -> dict:
    return {
        "capacity_base": "min_signal_trade",
        "volume_cap": 0.1,
        "buy_slippage": 0.0,
        "sell_slippage": 0.0,
        "commission_rate": 0.00025,
        "min_commission": 5.0,
        "transfer_fee_rate": 0.00001,
        "stamp_tax_rate_before_20230828": 0.001,
        "stamp_tax_rate_after_20230828": 0.0005,
        "impact_k": 0.0,
        "lot_size": 100,
        "limit_tolerance": 0.0001,
    }


def _row(**overrides) -> pd.Series:
    base = {
        "code": "000001",
        "open": 10.0,
        "high": 10.3,
        "low": 9.8,
        "close": 10.1,
        "pre_close": 10.0,
        "up_limit": 11.0,
        "down_limit": 9.0,
        "volume": 100000.0,
        "amount": 1e8,
        "paused": False,
    }
    base.update(overrides)
    return pd.Series(base)


def _position(entry_date: str = "2024-01-01") -> Position:
    return Position(
        code="000001",
        shares=1000,
        entry_date=pd.Timestamp(entry_date),
        entry_index=1,
        entry_price=10.0,
        entry_value=10000.0,
        entry_cost=5.0,
        highest_close=10.5,
        last_close=10.1,
        triggered=False,
    )


def test_can_buy_blocks_limit_up_open():
    cfg = _trade_cfg()
    row = _row(open=11.0, up_limit=11.0)
    ok, reason = can_buy(row, cfg)
    assert ok is False
    assert reason == "limit_up_open"


def test_can_buy_blocks_one_price_limit_up():
    cfg = _trade_cfg()
    row = _row(open=11.0, high=11.0, low=11.0, close=11.0, up_limit=11.0)
    ok, reason = can_buy(row, cfg)
    assert ok is False
    assert reason == "limit_up_open"


def test_can_sell_blocks_t_plus_1():
    cfg = _trade_cfg()
    row = _row()
    pos = _position(entry_date="2024-01-02")
    ok, reason = can_sell(row, pos, pd.Timestamp("2024-01-02"), cfg)
    assert ok is False
    assert reason == "t_plus_1"


def test_can_sell_blocks_limit_down_open():
    cfg = _trade_cfg()
    row = _row(open=9.0, down_limit=9.0)
    pos = _position(entry_date="2024-01-01")
    ok, reason = can_sell(row, pos, pd.Timestamp("2024-01-03"), cfg)
    assert ok is False
    assert reason == "limit_down_open"


def test_stamp_tax_switch_date_changes_sell_cost():
    cfg = _trade_cfg()
    value = 100000.0
    rate_before = stamp_tax_rate(pd.Timestamp("2023-08-25"), cfg)
    rate_after = stamp_tax_rate(pd.Timestamp("2023-08-28"), cfg)
    assert rate_before == 0.001
    assert rate_after == 0.0005

    sell_cost_before = trade_cost("sell", value, pd.Timestamp("2023-08-25"), cfg)
    sell_cost_after = trade_cost("sell", value, pd.Timestamp("2023-08-28"), cfg)
    assert sell_cost_after < sell_cost_before
