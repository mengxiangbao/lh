import pandas as pd

from dragon_backtest.backtester import compact_signal_output
from dragon_backtest.config import DEFAULT_CONFIG
from dragon_backtest.signals import build_signals


def _cfg() -> dict:
    cfg = {
        **DEFAULT_CONFIG,
        "universe": {**DEFAULT_CONFIG["universe"]},
        "signal": {**DEFAULT_CONFIG["signal"]},
        "score": {"weights": {**DEFAULT_CONFIG["score"]["weights"]}},
        "market_filter": {**DEFAULT_CONFIG.get("market_filter", {})},
        "market_sizing": {**DEFAULT_CONFIG.get("market_sizing", {})},
    }
    cfg["universe"]["min_list_days"] = 0
    cfg["universe"]["min_avg_amount_20d"] = 0.0
    cfg["universe"]["max_ret_20d"] = 9.99
    cfg["universe"]["max_limit_up_count_10d"] = 99
    cfg["universe"]["max_amount_ma5_to_ma60"] = 99.0
    cfg["universe"]["min_close_to_high_60d"] = 0.0
    cfg["universe"]["min_sector_shift_pct"] = -1.0
    cfg["universe"]["require_stock_stronger_than_sector"] = False
    cfg["universe"]["max_drawdown_vs_sector_median"] = False
    cfg["universe"]["min_up_amount_ratio_20d"] = 0.0
    cfg["signal"]["candidate_top_n"] = 10
    cfg["signal"]["direct_top_n"] = 10
    cfg["signal"]["trigger_amount_to_ma60"] = 0.0
    cfg["signal"]["trigger_ret_1d"] = -1.0
    cfg["signal"]["trigger_sector_ret_pct"] = -1.0
    cfg["signal"]["trigger_close_location"] = 0.0
    return cfg


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "code": "000001",
                "ts_code": "000001.SZ",
                "name": "A",
                "exchange": "SZ",
                "market": "主板",
                "sector": "Bank",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "pre_close": 10.0,
                "volume": 100000,
                "amount": 1e9,
                "float_mv": 1e10,
                "paused": False,
                "is_st": False,
                "list_days": 200,
                "amount_ma20": 1e9,
                "ret_20d": 0.05,
                "limit_up_count_10d": 0,
                "amount_ma5_to_ma60": 1.2,
                "close_to_high_60d": 0.95,
                "sector_shift_pct": 0.8,
                "up_amount_ratio_20d": 1.5,
                "rel_ret_20d": 0.02,
                "ret_20d_rank_in_sector": 0.1,
                "max_drawdown_20d": -0.05,
                "sector_max_drawdown_20d_median": -0.1,
                "high_60d_prev": 9.8,
                "amount_ma60": 5e8,
                "ret_1d": 0.06,
                "is_limit_up": False,
                "sector_ret_1d_pct": 0.9,
                "score_pct_all": 0.9,
                "hidden_rs_score": 0.8,
                "accumulation_score": 0.7,
                "vol_squeeze_score": 0.6,
                "volatility_energy_score": 0.5,
                "anti_fall_score": 0.7,
                "position_score": 0.6,
                "limit_gene_score": 0.5,
                "ret_5d": 0.03,
                "ret_10d": 0.04,
                "rel_ret_5d": 0.01,
                "sector_ret_20d_pct": 0.06,
                "market_ret_20d": 0.02,
                "market_amount_ratio": 1.1,
                "market_positive_ratio_ma5": 0.6,
                "market_limit_up_count": 10,
                "market_trend_regime": "uptrend",
                "market_liquidity_regime": "expanding",
                "market_breadth_regime": "strong_breadth",
                "market_combined_regime": "uptrend_strong",
                "future_max_return_20d": 0.5,
                "future_rank_in_sector_20d": 0.02,
                "days_to_start": 3,
                "y_start_10d": 1,
                "y_leader_20d": 1,
                "turnover": 0.02,
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
        ]
    )


def test_build_signals_does_not_require_future_label_columns():
    features = _features().drop(columns=["future_max_return_20d", "future_rank_in_sector_20d", "days_to_start", "y_start_10d", "y_leader_20d"])
    signals = build_signals(features, _cfg(), mode="confirmed")
    assert "buy_signal" in signals.columns
    assert bool(signals["buy_signal"].iloc[0]) is True


def test_compact_signal_output_excludes_future_columns_by_default():
    signals = build_signals(_features(), _cfg(), mode="confirmed")
    compact = compact_signal_output(signals)
    for col in ["future_max_return_20d", "future_rank_in_sector_20d", "days_to_start", "y_start_10d", "y_leader_20d"]:
        assert col not in compact.columns
