from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtester import run_backtest, write_backtest_outputs
from .alerts import build_alerts, write_alerts
from .config import load_config
from .data import load_daily, write_table
from .data_check import check_daily_data
from .data_fix import fix_daily_data
from .event_study import EventStudyConfig, run_event_study
from .features import prepare_features
from .labels import build_research_labels
from .minishare_source import fetch_minishare_mins, read_code_list
from .parameter_sweep import DEFAULT_SWEEP, parse_float_list, parse_int_list, run_parameter_sweep
from .performance import summarize_performance, write_metrics
from .regime_analysis import run_regime_analysis
from .reporting import write_report_tables
from .sample_data import generate_sample_daily
from .signals import build_signals
from .tushare_client import DEFAULT_TUSHARE_HTTP_URL
from .tushare_source import fetch_tushare_daily


def main() -> None:
    parser = argparse.ArgumentParser(description="A-share dragon leader potential backtester")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-sample", help="Generate synthetic A-share daily data")
    gen.add_argument("--out", default="data/raw", help="Output directory")
    gen.add_argument("--stocks", type=int, default=96)
    gen.add_argument("--days", type=int, default=520)
    gen.add_argument("--seed", type=int, default=7)

    fetch = sub.add_parser("fetch-tushare", help="Fetch and normalize A-share daily data from Tushare")
    fetch.add_argument("--start", required=True, help="Start date, e.g. 20200101")
    fetch.add_argument("--end", required=True, help="End date, e.g. 20241231")
    fetch.add_argument("--out", default="data/raw/daily_price.csv")
    fetch.add_argument("--token", default=None, help="Prefer env var TUSHARE_TOKEN instead of this option")
    fetch.add_argument("--http-url", default=DEFAULT_TUSHARE_HTTP_URL)
    fetch.add_argument("--sleep", type=float, default=0.12, help="Sleep seconds between API calls")
    fetch.add_argument("--skip-namechange", action="store_true", help="Skip historical ST namechange lookup")
    fetch.add_argument("--cache-dir", default=None, help="Optional per-trade-date cache directory for resumable fetches")
    fetch.add_argument("--retries", type=int, default=3, help="Retry count for each Tushare endpoint call")
    fetch.add_argument("--retry-sleep", type=float, default=2.0, help="Seconds to sleep between retries")

    mins = sub.add_parser("fetch-minishare-mins", help="Fetch historical minute bars from minishare")
    mins.add_argument("--codes", default=None, help="Comma separated ts_code list, e.g. 600000.SH,000001.SZ")
    mins.add_argument("--codes-file", default=None, help="CSV/TXT file with ts_code or code column")
    mins.add_argument("--freq", choices=["5min", "15min", "30min", "60min"], default="5min")
    mins.add_argument("--start", required=True, help='Start datetime, e.g. "20250210 09:00:00"')
    mins.add_argument("--end", required=True, help='End datetime, e.g. "20250210 19:00:00"')
    mins.add_argument("--out-dir", default="data/raw/minute")
    mins.add_argument("--token", default=None, help="Prefer env var MINISHARE_TOKEN instead of this option")
    mins.add_argument("--sleep", type=float, default=0.05, help="Sleep seconds between API calls")
    mins.add_argument("--combine", action="store_true", help="Also write one combined CSV")

    check = sub.add_parser("check-data", help="Check daily data quality before backtest")
    check.add_argument("--data", default="data/raw/daily_price.csv", help="Daily CSV/Parquet path")
    check.add_argument("--out", default="data/data_check", help="Output report directory")
    check.add_argument("--sample-rows", type=int, default=10)

    fix = sub.add_parser("fix-daily", help="Fix common daily data issues into a new file")
    fix.add_argument("--input", default="data/raw/daily_price.csv", help="Input daily CSV")
    fix.add_argument("--out", default="data/raw/daily_price_clean.csv", help="Output cleaned daily CSV")

    alerts = sub.add_parser("build-alerts", help="Build daily alert table from backtest signals")
    alerts.add_argument("--signals", default="data/backtest_result/confirmed/signals.csv")
    alerts.add_argument("--out", default="data/backtest_result/confirmed")
    alerts.add_argument("--top-n", type=int, default=30)
    alerts.add_argument("--no-watch", action="store_true", help="Exclude watch-only candidates")

    sweep = sub.add_parser("sweep-params", help="Run parameter stability sweep")
    sweep.add_argument("--config", default="config/default.toml")
    sweep.add_argument("--data", default="data/raw/daily_price.csv")
    sweep.add_argument("--out", default="data/param_sweep")
    sweep.add_argument("--mode", choices=["potential", "confirmed", "hybrid"], default="confirmed")
    sweep.add_argument("--start", default=None)
    sweep.add_argument("--end", default=None)
    sweep.add_argument("--no-labels", action="store_true")
    sweep.add_argument("--candidate-top-n", default=None, help="Comma list, e.g. 30,50")
    sweep.add_argument("--trigger-volume", default=None, help="Comma list, e.g. 1.3,1.5,2.0")
    sweep.add_argument("--stop-loss", default=None, help="Comma list, e.g. -0.08,-0.10")
    sweep.add_argument("--max-holding-days", default=None, help="Comma list, e.g. 20,30")

    reports = sub.add_parser("build-reports", help="Build yearly/monthly performance reports from backtest outputs")
    reports.add_argument("--input", default="data/backtest_result/confirmed", help="Directory with equity/trades/positions CSV")
    reports.add_argument("--out", default=None, help="Output directory, defaults to --input")

    summarize = sub.add_parser("summarize-backtest", help="Build metrics, reports, and alerts from existing backtest CSV outputs")
    summarize.add_argument("--input", default="data/backtest_result/confirmed", help="Directory with equity/trades/signals CSV")
    summarize.add_argument("--config", default="config/default.toml")
    summarize.add_argument("--out", default=None, help="Output directory, defaults to --input")

    regimes = sub.add_parser("analyze-regimes", help="Analyze time slices and market regimes for a backtest")
    regimes.add_argument("--data", default="data/raw/daily_price.csv", help="Daily data path")
    regimes.add_argument("--input", default="data/backtest_result/event_tuned", help="Directory with backtest outputs")
    regimes.add_argument("--out", default=None, help="Output directory, defaults to --input")

    study = sub.add_parser("study-start-events", help="Study early features of successful and failed launch events")
    study.add_argument("--config", default="config/default.toml")
    study.add_argument("--data", default="data/raw/daily_price.csv")
    study.add_argument("--out", default="data/event_study")
    study.add_argument("--start", default=None)
    study.add_argument("--end", default=None)
    study.add_argument("--pre-window", type=int, default=20)
    study.add_argument("--post-window", type=int, default=20)
    study.add_argument("--min-gap", type=int, default=20)
    study.add_argument("--trigger-volume", type=float, default=1.5)
    study.add_argument("--trigger-ret", type=float, default=0.05)
    study.add_argument("--sector-shift-pct", type=float, default=0.70)
    study.add_argument("--future-return-min", type=float, default=0.25)
    study.add_argument("--future-rank-max", type=float, default=0.10)
    study.add_argument("--fail-return-max", type=float, default=0.10)
    study.add_argument("--fail-rank-min", type=float, default=0.30)
    study.add_argument("--save-window", action="store_true", help="Also save event_window.csv")

    bt = sub.add_parser("backtest", help="Run daily rule-score backtest")
    bt.add_argument("--config", default="config/default.toml")
    bt.add_argument("--data", default=None, help="Daily data path, overrides config data.daily_path")
    bt.add_argument("--out", default="data/backtest_result/confirmed")
    bt.add_argument("--mode", choices=["potential", "confirmed", "hybrid"], default=None)
    bt.add_argument("--start", default=None)
    bt.add_argument("--end", default=None)
    bt.add_argument("--no-labels", action="store_true", help="Skip research label metrics")
    bt.add_argument("--save-features", action="store_true")

    args = parser.parse_args()
    if args.command == "generate-sample":
        path = generate_sample_daily(args.out, args.stocks, args.days, args.seed)
        print(f"sample data written: {path}")
        return

    if args.command == "fetch-tushare":
        path = fetch_tushare_daily(
            start=args.start,
            end=args.end,
            out_path=args.out,
            token=args.token,
            http_url=args.http_url,
            sleep_seconds=args.sleep,
            skip_namechange=args.skip_namechange,
            cache_dir=args.cache_dir,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        print(f"tushare daily data written: {path}")
        return

    if args.command == "fetch-minishare-mins":
        codes = read_code_list(args.codes, args.codes_file)
        paths = fetch_minishare_mins(
            codes=codes,
            freq=args.freq,
            start=args.start,
            end=args.end,
            out_dir=args.out_dir,
            token=args.token,
            sleep_seconds=args.sleep,
            combine=args.combine,
        )
        print(f"minishare minute files written: {len(paths)}")
        for path in paths[:10]:
            print(f"  {path}")
        if len(paths) > 10:
            print(f"  ... {len(paths) - 10} more")
        return

    if args.command == "check-data":
        report = check_daily_data(args.data, args.out, args.sample_rows)
        counts = report["severity_counts"]
        print(f"data rows: {report['rows']}, stocks: {report['stock_count']}, range: {report['start_date']} ~ {report['end_date']}")
        print(f"errors={counts['error']}, warnings={counts['warning']}, info={counts['info']}")
        print(f"can_backtest={report['can_backtest']}")
        print(f"report written: {args.out}")
        if report["issues"]:
            print("top issues:")
            for issue in report["issues"][:8]:
                print(f"  [{issue['severity']}] {issue['check']}: {issue['message']} rows={issue['rows']}")
        return

    if args.command == "fix-daily":
        report = fix_daily_data(args.input, args.out)
        print(f"cleaned daily written: {args.out}")
        print(
            f"rows={report['rows']}, stocks={report['stock_count']}, "
            f"range={report['start_date']} ~ {report['end_date']}, "
            f"fixed_list_date_rows={report['fixed_list_date_rows']}, "
            f"fixed_sector_rows={report['fixed_sector_rows']}"
        )
        return

    if args.command == "build-alerts":
        signals = pd.read_csv(args.signals, parse_dates=["date"])
        alert_df = build_alerts(signals, top_n=args.top_n, include_watch=not args.no_watch)
        write_alerts(alert_df, args.out)
        print(f"alerts written: {args.out}")
        print(f"rows={len(alert_df)}")
        return

    if args.command == "sweep-params":
        grid = {
            "candidate_top_n": parse_int_list(args.candidate_top_n, DEFAULT_SWEEP["candidate_top_n"]),
            "trigger_amount_to_ma60": parse_float_list(args.trigger_volume, DEFAULT_SWEEP["trigger_amount_to_ma60"]),
            "stop_loss": parse_float_list(args.stop_loss, DEFAULT_SWEEP["stop_loss"]),
            "max_holding_days": parse_int_list(args.max_holding_days, DEFAULT_SWEEP["max_holding_days"]),
        }
        result_df = run_parameter_sweep(
            data_path=args.data,
            config_path=args.config,
            out_dir=args.out,
            mode=args.mode,
            start=args.start,
            end=args.end,
            include_labels=not args.no_labels,
            grid=grid,
        )
        print(f"sweep written: {args.out}")
        print(f"runs={len(result_df)}")
        if not result_df.empty:
            display_cols = [
                "run_id",
                "candidate_top_n",
                "trigger_amount_to_ma60",
                "stop_loss",
                "max_holding_days",
                "total_return",
                "max_drawdown",
                "sharpe",
                "round_trip_count",
            ]
            print(result_df[display_cols].head(10).to_string(index=False))
        return

    if args.command == "build-reports":
        input_dir = Path(args.input)
        out_dir = Path(args.out) if args.out else input_dir
        result = {
            "equity": pd.read_csv(input_dir / "equity.csv", parse_dates=["date"]),
            "trades": pd.read_csv(input_dir / "trades.csv", parse_dates=["trade_date", "signal_date"]),
            "positions": pd.read_csv(input_dir / "positions.csv", parse_dates=["date", "entry_date"]),
        }
        write_report_tables(result, out_dir)
        print(f"reports written: {out_dir}")
        return

    if args.command == "summarize-backtest":
        input_dir = Path(args.input)
        out_dir = Path(args.out) if args.out else input_dir
        cfg = load_config(args.config)
        result = {
            "equity": pd.read_csv(input_dir / "equity.csv", parse_dates=["date"]),
            "trades": pd.read_csv(input_dir / "trades.csv", parse_dates=["trade_date", "signal_date"]),
            "positions": pd.read_csv(input_dir / "positions.csv", parse_dates=["date", "entry_date"])
            if (input_dir / "positions.csv").exists()
            else pd.DataFrame(),
            "signals": pd.read_csv(input_dir / "signals.csv", parse_dates=["date"])
            if (input_dir / "signals.csv").exists()
            else pd.DataFrame(),
        }
        metrics = summarize_performance(result, cfg)
        write_metrics(metrics, out_dir)
        write_report_tables(result, out_dir)
        if not result["signals"].empty:
            write_alerts(build_alerts(result["signals"]), out_dir)
        print(f"summary outputs written: {out_dir}")
        print(
            "summary: "
            f"return={metrics.get('total_return', 0):.2%}, "
            f"mdd={metrics.get('max_drawdown', 0):.2%}, "
            f"sharpe={metrics.get('sharpe', 0):.2f}, "
            f"filled_buys={metrics.get('filled_buy_count', 0)}"
        )
        return

    if args.command == "analyze-regimes":
        out_dir = Path(args.out) if args.out else Path(args.input)
        tables = run_regime_analysis(args.data, args.input, out_dir)
        print(f"regime analysis written: {out_dir}")
        monthly = tables.get("monthly_contribution", pd.DataFrame())
        regimes_df = tables.get("market_regime_performance", pd.DataFrame())
        if not monthly.empty:
            positive = int((monthly["total_return"] > 0).sum())
            print(f"monthly positive={positive}/{len(monthly)}")
            print(monthly.sort_values("total_return", ascending=False).head(5).to_string(index=False))
        if not regimes_df.empty:
            print(regimes_df.sort_values("total_return", ascending=False).head(8).to_string(index=False))
        return

    if args.command == "study-start-events":
        study_cfg = EventStudyConfig(
            pre_window=args.pre_window,
            post_window=args.post_window,
            min_gap=args.min_gap,
            trigger_volume=args.trigger_volume,
            trigger_ret=args.trigger_ret,
            sector_shift_pct=args.sector_shift_pct,
            future_return_min=args.future_return_min,
            future_rank_max=args.future_rank_max,
            fail_return_max=args.fail_return_max,
            fail_rank_min=args.fail_rank_min,
        )
        tables = run_event_study(
            data_path=args.data,
            config_path=args.config,
            out_dir=args.out,
            start=args.start,
            end=args.end,
            study_cfg=study_cfg,
            save_window=args.save_window,
        )
        print(f"event study written: {args.out}")
        summary = tables.get("event_summary", pd.DataFrame())
        if not summary.empty:
            print(summary.to_string(index=False))
        return

    if args.command == "backtest":
        cfg = load_config(args.config)
        data_path = args.data or cfg["data"]["daily_path"]
        daily = load_daily(data_path, args.start, args.end)
        features = prepare_features(daily)
        if not args.no_labels:
            features = build_research_labels(features)
        signals = build_signals(features, cfg, args.mode)
        result = run_backtest(signals, cfg, args.mode)
        metrics = summarize_performance(result, cfg)
        out_dir = Path(args.out)
        write_backtest_outputs(result, out_dir)
        write_metrics(metrics, out_dir)
        if args.save_features:
            write_table(features, out_dir / "features.csv")
        print(f"outputs written: {out_dir}")
        print(
            "summary: "
            f"return={metrics.get('total_return', 0):.2%}, "
            f"mdd={metrics.get('max_drawdown', 0):.2%}, "
            f"sharpe={metrics.get('sharpe', 0):.2f}, "
            f"filled_buys={metrics.get('filled_buy_count', 0)}"
        )
