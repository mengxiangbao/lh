from copy import deepcopy

from dragon_backtest.config import DEFAULT_CONFIG, validate_config


def test_invalid_trade_volume_cap_raises():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["trade"]["volume_cap"] = 1.5
    try:
        validate_config(cfg)
        raise AssertionError("Expected ValueError for invalid trade.volume_cap")
    except ValueError as exc:
        assert "trade.volume_cap" in str(exc)


def test_invalid_capacity_base_raises():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["trade"]["capacity_base"] = "bad_mode"
    try:
        validate_config(cfg)
        raise AssertionError("Expected ValueError for invalid trade.capacity_base")
    except ValueError as exc:
        assert "trade.capacity_base" in str(exc)


def test_invalid_stop_loss_raises():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["risk"]["stop_loss"] = 0.01
    try:
        validate_config(cfg)
        raise AssertionError("Expected ValueError for invalid risk.stop_loss")
    except ValueError as exc:
        assert "risk.stop_loss" in str(exc)

