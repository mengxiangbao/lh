from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .alerts import build_alerts, write_alerts
from .broker import Position, execute_buy, execute_sell
from .reporting import write_report_tables
from .signals import select_buy_rows


SIGNAL_OUTPUT_COLUMNS = [
    "date",
    "ts_code",
    "code",
    "name",
    "exchange",
    "market",
    "sector",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "amount",
    "float_mv",
    "score",
    "score_pct_all",
    "candidate_rank",
    "candidate",
    "pre_pool",
    "trigger",
    "buy_signal",
    "raw_buy_signal",
    "market_filter_ok",
    "market_size_mult",
    "target_weight_mult",
    "signal_mode",
    "trigger_breakout_60d",
    "trigger_volume",
    "trigger_strong_day",
    "trigger_sector",
    "trigger_close_location",
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "rel_ret_5d",
    "rel_ret_20d",
    "ret_20d_rank_in_sector",
    "sector_ret_1d_pct",
    "sector_ret_20d_pct",
    "sector_shift_pct",
    "hidden_rs_score",
    "accumulation_score",
    "vol_squeeze_score",
    "volatility_energy_score",
    "anti_fall_score",
    "position_score",
    "limit_gene_score",
    "close_to_high_60d",
    "close_position_20d",
    "amount_ma5_to_ma60",
    "up_amount_ratio_20d",
    "limit_up_count_10d",
    "market_ret_20d",
    "market_amount_ratio",
    "market_positive_ratio_ma5",
    "market_limit_up_count",
    "market_trend_regime",
    "market_liquidity_regime",
    "market_breadth_regime",
    "market_combined_regime",
]


def run_backtest(signals: pd.DataFrame, cfg: dict, mode: str | None = None) -> dict:
    mode = mode or cfg["signal"]["mode"]
    trade_cfg = cfg["trade"]
    risk_cfg = cfg["risk"]

    dates = list(pd.Index(signals["date"].drop_duplicates()).sort_values())
    date_to_idx = {date: i for i, date in enumerate(dates)}
    by_date = {date: day.set_index("code", drop=False) for date, day in signals.groupby("date", sort=True)}

    cash = float(trade_cfg["initial_cash"])
    last_equity = cash
    positions: dict[str, Position] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []
    position_rows: list[dict] = []

    for i in range(1, len(dates)):
        signal_date = dates[i - 1]
        trade_date = dates[i]
        signal_day = by_date[signal_date]
        trade_day = by_date[trade_date]

        sell_reasons = build_sell_orders(signal_day, positions, signal_date, date_to_idx, risk_cfg, cfg.get("market_filter", {}))
        sell_codes = set(sell_reasons)
        slots = max(int(trade_cfg["max_positions"]) - (len(positions) - len(sell_codes)), 0)
        buy_rows = select_buy_rows(signal_day, set(positions), sell_codes, slots)

        for code, reason in sell_reasons.items():
            if code not in positions:
                continue
            if code not in trade_day.index:
                trades.append(blocked_trade(trade_date, code, "sell", "missing_trade_row", reason, cash))
                continue
            filled, cash, record = execute_sell(trade_day.loc[code], positions[code], trade_date, cash, trade_cfg, reason)
            record["holding_days"] = date_to_idx[trade_date] - positions[code].entry_index
            trades.append(record)
            if filled:
                del positions[code]

        for _, signal_row in buy_rows.iterrows():
            code = str(signal_row["code"])
            if code in positions:
                continue
            if len(positions) >= trade_cfg["max_positions"]:
                break
            if code not in trade_day.index:
                trades.append(blocked_trade(trade_date, code, "buy", "missing_trade_row", "entry", cash))
                continue
            target_value = last_equity * trade_cfg["target_weight"] * float(signal_row.get("target_weight_mult", 1.0))
            position, cash, record = execute_buy(
                trade_day.loc[code],
                signal_row,
                trade_date,
                date_to_idx[trade_date],
                target_value,
                cash,
                trade_cfg,
            )
            trades.append(record)
            if position is not None:
                positions[code] = position

        market_value = 0.0
        for code, position in list(positions.items()):
            if code in trade_day.index:
                row = trade_day.loc[code]
                close = float(row["close"])
                position.last_close = close
                position.highest_close = max(position.highest_close, close)
                position.triggered = position.triggered or bool(row.get("trigger", False))
                price_for_value = close
                score = float(row.get("score", 0.0))
                sector = row.get("sector", "")
                ts_code = row.get("ts_code", code)
                name = row.get("name", "")
                exchange = row.get("exchange", "")
                market = row.get("market", "")
            else:
                price_for_value = position.last_close
                score = 0.0
                sector = ""
                ts_code = code
                name = ""
                exchange = ""
                market = ""
            value = position.shares * price_for_value
            market_value += value
            position_rows.append(
                {
                    "date": trade_date,
                    "ts_code": ts_code,
                    "code": code,
                    "name": name,
                    "exchange": exchange,
                    "market": market,
                    "sector": sector,
                    "shares": position.shares,
                    "close": price_for_value,
                    "market_value": value,
                    "entry_date": position.entry_date,
                    "entry_price": position.entry_price,
                    "highest_close": position.highest_close,
                    "triggered": position.triggered,
                    "score": score,
                }
            )

        total_equity = cash + market_value
        equity_rows.append(
            {
                "date": trade_date,
                "cash": cash,
                "market_value": market_value,
                "total_equity": total_equity,
                "positions": len(positions),
                "daily_return": total_equity / last_equity - 1 if last_equity else 0.0,
                "mode": mode,
            }
        )
        last_equity = total_equity

    return {
        "equity": pd.DataFrame(equity_rows),
        "trades": pd.DataFrame(trades),
        "positions": pd.DataFrame(position_rows),
        "signals": signals,
    }


def build_sell_orders(
    signal_day: pd.DataFrame,
    positions: dict[str, Position],
    signal_date: pd.Timestamp,
    date_to_idx: dict[pd.Timestamp, int],
    risk_cfg: dict,
    market_filter_cfg: dict | None = None,
) -> dict[str, str]:
    sell: dict[str, str] = {}
    signal_idx = date_to_idx[signal_date]
    for code, pos in positions.items():
        if code not in signal_day.index:
            sell[code] = "missing_signal_row"
            continue
        row = signal_day.loc[code]
        close = float(row["close"])
        holding_days = signal_idx - pos.entry_index
        pos.triggered = pos.triggered or bool(row.get("trigger", False))
        pos.highest_close = max(pos.highest_close, close)
        pos.last_close = close

        if holding_days >= risk_cfg["max_holding_days"]:
            sell[code] = "max_holding_days"
        elif close / pos.entry_price - 1 <= risk_cfg["stop_loss"]:
            sell[code] = "stop_loss"
        elif close / max(pos.highest_close, pos.entry_price) - 1 <= risk_cfg["trailing_stop"]:
            sell[code] = "trailing_stop"
        elif market_filter_cfg and market_filter_cfg.get("exit_on_fail", False) and not bool(row.get("market_filter_ok", True)):
            sell[code] = "market_filter_exit"
        elif risk_cfg["exit_ma"] in row and close < row[risk_cfg["exit_ma"]]:
            sell[code] = f"break_{risk_cfg['exit_ma']}"
        elif (not pos.triggered) and holding_days >= risk_cfg["stale_exit_days"]:
            sell[code] = "stale_no_trigger"
        elif row.get("sector_shift_pct", 1.0) < risk_cfg["sector_weak_pct"]:
            sell[code] = "sector_weak"
    return sell


def blocked_trade(trade_date, code: str, side: str, blocked_reason: str, reason: str, cash: float) -> dict:
    return {
        "trade_date": trade_date,
        "signal_date": pd.NaT,
        "code": code,
        "side": side,
        "status": "blocked",
        "blocked_reason": blocked_reason,
        "reason": reason,
        "shares": 0,
        "price": 0.0,
        "value": 0.0,
        "cost": 0.0,
        "cash_after": cash,
        "score": 0.0,
        "trigger": False,
    }


def write_backtest_outputs(result: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["equity", "trades", "positions"]:
        df = result.get(name)
        if isinstance(df, pd.DataFrame):
            df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    signals = result.get("signals")
    if isinstance(signals, pd.DataFrame):
        compact_signal_output(signals).to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
        write_alerts(build_alerts(signals), out_dir)
    write_report_tables(result, out_dir)


def compact_signal_output(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals

    mask = pd.Series(False, index=signals.index)
    for col in ["candidate", "pre_pool", "trigger", "buy_signal"]:
        if col in signals.columns:
            mask |= signals[col].fillna(False).astype(bool)
    rows = signals.loc[mask].copy() if mask.any() else signals.copy()
    cols = [col for col in SIGNAL_OUTPUT_COLUMNS if col in rows.columns]
    return rows[cols].sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)
