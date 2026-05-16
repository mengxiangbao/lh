from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dragon_backtest.backtester import run_backtest, write_backtest_outputs
from dragon_backtest.config import load_config
from dragon_backtest.data import load_daily, write_table
from dragon_backtest.data_check import check_daily_data
from dragon_backtest.features import prepare_features
from dragon_backtest.labels import build_research_labels
from dragon_backtest.parameter_sweep import (
    DEFAULT_SWEEP,
    parse_float_list,
    parse_int_list,
    run_parameter_sweep,
)
from dragon_backtest.performance import summarize_performance, write_metrics
from dragon_backtest.signals import build_signals


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config/event_tuned.toml"
DEFAULT_DATA_ROOT = ROOT / "data"


def default_daily_path(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    try:
        cfg = load_config(config_path)
        candidate = Path(cfg.get("data", {}).get("daily_path", "data/raw/daily_price_long_clean.csv"))
    except Exception:
        candidate = Path("data/raw/daily_price_long_clean.csv")
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv_if_exists(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "-"


def number(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def run_data_check(data_path: Path, out_dir: Path) -> dict:
    return check_daily_data(data_path, out_dir)


def run_research_backtest(
    data_path: Path,
    config_path: Path,
    out_dir: Path,
    mode: str,
    start: str | None,
    end: str | None,
    include_labels: bool,
    save_features: bool,
) -> dict:
    cfg = load_config(config_path)
    daily = load_daily(data_path, start or None, end or None)
    features = prepare_features(daily)
    if include_labels:
        features = build_research_labels(features)
    signals = build_signals(features, cfg, mode)
    result = run_backtest(signals, cfg, mode)
    metrics = summarize_performance(result, cfg)
    write_backtest_outputs(result, out_dir)
    write_metrics(metrics, out_dir)
    if save_features:
        write_table(features, out_dir / "features.csv")
    return metrics


def render_sidebar():
    st.sidebar.header("运行参数")
    data_path = st.sidebar.text_input("日线数据路径", str(default_daily_path()))
    config_path = st.sidebar.text_input("配置文件路径", "config/event_tuned.toml")
    mode = st.sidebar.selectbox("回测模式", ["confirmed", "potential", "hybrid"], index=0)
    out_dir = st.sidebar.text_input("结果输出目录", str(DEFAULT_DATA_ROOT / f"backtest_result/{mode}"))
    check_dir = st.sidebar.text_input("数据检查目录", str(DEFAULT_DATA_ROOT / "data_check"))

    st.sidebar.divider()
    start = st.sidebar.text_input("开始日期，可空", "")
    end = st.sidebar.text_input("结束日期，可空", "")
    include_labels = st.sidebar.checkbox("生成研究标签指标", value=True)
    save_features = st.sidebar.checkbox("保存 features.csv", value=False)

    return {
        "data_path": rel_path(data_path),
        "config_path": rel_path(config_path),
        "mode": mode,
        "out_dir": rel_path(out_dir),
        "check_dir": rel_path(check_dir),
        "start": start.strip(),
        "end": end.strip(),
        "include_labels": include_labels,
        "save_features": save_features,
    }


def render_metric_overview(metrics: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("总收益", pct(metrics.get("total_return")))
    cols[1].metric("年化收益", pct(metrics.get("annual_return")))
    cols[2].metric("最大回撤", pct(metrics.get("max_drawdown")))
    cols[3].metric("夏普", number(metrics.get("sharpe")))
    cols[4].metric("Calmar", number(metrics.get("calmar")))

    cols = st.columns(5)
    cols[0].metric("胜率", pct(metrics.get("win_rate")))
    cols[1].metric("单笔均值", pct(metrics.get("avg_trade_return")))
    cols[2].metric("买入成交率", pct(metrics.get("tradable_buy_rate")))
    cols[3].metric("涨停不可买", str(metrics.get("limit_buy_block_count", 0)))
    cols[4].metric("跌停不可卖", str(metrics.get("limit_sell_block_count", 0)))

    top_cols = st.columns(4)
    top_cols[0].metric("Top10启动命中", pct(metrics.get("top10_precision_start_10d")))
    top_cols[1].metric("Top10龙头命中", pct(metrics.get("top10_leader_hit_rate_20d")))
    top_cols[2].metric("Top10未来最大涨幅", pct(metrics.get("top10_avg_future_max_return_20d")))
    top_cols[3].metric("平均提前天数", number(metrics.get("top10_avg_lead_days")))


def render_equity(out_dir: Path) -> None:
    equity = read_csv_if_exists(out_dir / "equity.csv", ["date"])
    if equity.empty:
        st.info("还没有权益曲线。先运行一次回测。")
        return
    chart_df = equity.set_index("date")[["total_equity"]]
    st.line_chart(chart_df, height=320)

    drawdown = chart_df["total_equity"] / chart_df["total_equity"].cummax() - 1
    st.caption("回撤曲线")
    st.line_chart(drawdown.rename("drawdown"), height=220)
    st.dataframe(equity.tail(30), width="stretch")


def render_trades(out_dir: Path) -> None:
    trades = read_csv_if_exists(out_dir / "trades.csv", ["trade_date", "signal_date"])
    if trades.empty:
        st.info("还没有交易明细。")
        return
    side = st.radio("交易方向", ["全部", "买入", "卖出"], horizontal=True)
    view = trades
    if side == "买入":
        view = trades[trades["side"] == "buy"]
    elif side == "卖出":
        view = trades[trades["side"] == "sell"]
    preferred = [
        "trade_date",
        "signal_date",
        "ts_code",
        "code",
        "name",
        "exchange",
        "market",
        "sector",
        "side",
        "status",
        "reason",
        "blocked_reason",
        "shares",
        "price",
        "value",
        "cost",
        "pnl",
        "return",
        "holding_days",
        "score",
        "trigger",
    ]
    view_cols = [col for col in preferred if col in view.columns] + [col for col in view.columns if col not in preferred]
    view = view[view_cols]
    st.dataframe(view.sort_values("trade_date", ascending=False), width="stretch")


def render_signals(out_dir: Path) -> None:
    signals = read_csv_if_exists(out_dir / "signals.csv", ["date"])
    if signals.empty:
        st.info("还没有信号表。")
        return
    dates = sorted(signals["date"].dropna().dt.strftime("%Y-%m-%d").unique(), reverse=True)
    selected = st.selectbox("交易日", dates, index=0)
    day = signals[signals["date"].dt.strftime("%Y-%m-%d") == selected].copy()
    only = st.radio("显示范围", ["候选池", "买入信号", "全部"], horizontal=True)
    if only == "候选池":
        day = day[day.get("candidate", False).astype(bool)]
    elif only == "买入信号":
        day = day[day.get("buy_signal", False).astype(bool)]
    cols = [
        "date",
        "ts_code",
        "code",
        "name",
        "exchange",
        "market",
        "sector",
        "score",
        "candidate_rank",
        "candidate",
        "pre_pool",
        "trigger",
        "buy_signal",
        "ret_20d",
        "rel_ret_20d",
        "close_to_high_60d",
        "amount_ma5_to_ma60",
        "sector_shift_pct",
        "hidden_rs_score",
        "accumulation_score",
        "vol_squeeze_score",
        "anti_fall_score",
        "position_score",
    ]
    cols = [col for col in cols if col in day.columns]
    st.dataframe(day.sort_values("score", ascending=False)[cols], width="stretch")


def render_alerts(out_dir: Path) -> None:
    alerts = read_csv_if_exists(out_dir / "alerts.csv", ["date"])
    if alerts.empty:
        st.info("还没有每日预警表。运行回测后会自动生成 alerts.csv。")
        return

    dates = sorted(alerts["date"].dropna().dt.strftime("%Y-%m-%d").unique(), reverse=True)
    actions = ["全部"] + sorted(alerts["action"].dropna().unique().tolist())
    cols = st.columns([2, 2, 3])
    selected_date = cols[0].selectbox("预警日期", dates, index=0)
    selected_action = cols[1].selectbox("建议动作", actions, index=0)

    day = alerts[alerts["date"].dt.strftime("%Y-%m-%d") == selected_date].copy()
    if selected_action != "全部":
        day = day[day["action"] == selected_action]

    latest_path = out_dir / "alerts_latest.csv"
    if latest_path.exists():
        cols[2].download_button(
            "下载最新预警表",
            data=latest_path.read_bytes(),
            file_name="alerts_latest.csv",
            mime="text/csv",
            width="stretch",
        )

    show_cols = [
        "date",
        "ts_code",
        "code",
        "name",
        "exchange",
        "market",
        "sector",
        "action",
        "reason",
        "score",
        "candidate_rank",
        "sector_shift_pct",
        "hidden_rs_score",
        "accumulation_score",
        "vol_squeeze_score",
        "anti_fall_score",
        "position_score",
        "ret_1d",
        "ret_20d",
        "rel_ret_20d",
        "close_to_high_60d",
        "amount_ma5_to_ma60",
        "up_amount_ratio_20d",
        "close",
        "amount",
        "float_mv",
    ]
    show_cols = [col for col in show_cols if col in day.columns]
    st.dataframe(day[show_cols].sort_values(["action", "score"], ascending=[True, False]), width="stretch")


def render_parameter_sweep(params: dict) -> None:
    st.subheader("参数稳定性测试")
    st.caption("同一份特征只计算一次，再批量替换关键参数。网格不要一次开太大。")

    cols = st.columns(4)
    candidate_top_n = cols[0].text_input("候选数量", "30,50")
    trigger_volume = cols[1].text_input("放量阈值", "1.3,1.5")
    stop_loss = cols[2].text_input("止损", "-0.08,-0.10")
    max_holding_days = cols[3].text_input("最长持仓", "20,30")

    out_dir_text = st.text_input("参数测试输出目录", str(DEFAULT_DATA_ROOT / "param_sweep"))
    sweep_out_dir = rel_path(out_dir_text)

    if st.button("运行参数测试", type="primary"):
        grid = {
            "candidate_top_n": parse_int_list(candidate_top_n, DEFAULT_SWEEP["candidate_top_n"]),
            "trigger_amount_to_ma60": parse_float_list(trigger_volume, DEFAULT_SWEEP["trigger_amount_to_ma60"]),
            "stop_loss": parse_float_list(stop_loss, DEFAULT_SWEEP["stop_loss"]),
            "max_holding_days": parse_int_list(max_holding_days, DEFAULT_SWEEP["max_holding_days"]),
        }
        run_count = 1
        for values in grid.values():
            run_count *= len(values)
        with st.spinner(f"正在运行 {run_count} 组参数..."):
            try:
                result_df = run_parameter_sweep(
                    data_path=params["data_path"],
                    config_path=params["config_path"],
                    out_dir=sweep_out_dir,
                    mode=params["mode"],
                    start=params["start"],
                    end=params["end"],
                    include_labels=params["include_labels"],
                    grid=grid,
                )
                st.success(f"参数测试完成，共 {len(result_df)} 组。")
            except Exception as exc:
                st.exception(exc)

    result_path = sweep_out_dir / "sweep_results.csv"
    result_df = read_csv_if_exists(result_path)
    if result_df.empty:
        st.info("还没有参数测试结果。")
        return

    metric_cols = [
        "run_id",
        "candidate_top_n",
        "trigger_amount_to_ma60",
        "stop_loss",
        "max_holding_days",
        "total_return",
        "annual_return",
        "max_drawdown",
        "sharpe",
        "calmar",
        "round_trip_count",
        "win_rate",
        "tradable_buy_rate",
        "top10_precision_start_10d",
        "top10_leader_hit_rate_20d",
        "return_to_drawdown",
    ]
    metric_cols = [col for col in metric_cols if col in result_df.columns]
    st.dataframe(result_df[metric_cols], width="stretch")
    st.download_button(
        "下载参数测试结果",
        data=result_path.read_bytes(),
        file_name="sweep_results.csv",
        mime="text/csv",
        width="stretch",
    )

    chart_cols = [col for col in ["total_return", "max_drawdown", "sharpe", "round_trip_count"] if col in result_df.columns]
    if chart_cols:
        chart_df = result_df.set_index("run_id")[chart_cols]
        st.line_chart(chart_df, height=280)


def render_performance_reports(out_dir: Path) -> None:
    st.subheader("分年/月度绩效报告")

    yearly = read_csv_if_exists(out_dir / "yearly_performance.csv")
    monthly = read_csv_if_exists(out_dir / "monthly_returns.csv")
    drawdowns = read_csv_if_exists(out_dir / "drawdown_periods.csv")
    trade_dist = read_csv_if_exists(out_dir / "trade_distribution.csv")
    sector = read_csv_if_exists(out_dir / "sector_exposure.csv")

    if yearly.empty and monthly.empty and drawdowns.empty:
        st.info("还没有绩效报告。重新运行一次回测后会自动生成。")
        return

    report_tabs = st.tabs(["年度绩效", "月度收益", "主要回撤", "交易分布", "板块暴露"])

    with report_tabs[0]:
        if yearly.empty:
            st.info("暂无年度绩效。")
        else:
            st.dataframe(yearly, width="stretch")

    with report_tabs[1]:
        if monthly.empty:
            st.info("暂无月度收益。")
        else:
            st.dataframe(monthly, width="stretch")
            chart_df = monthly.set_index("year")[[col for col in monthly.columns if col.endswith("total") or col == "year_total"]]
            if not chart_df.empty:
                st.bar_chart(chart_df, height=260)

    with report_tabs[2]:
        if drawdowns.empty:
            st.success("暂无显著回撤区间。")
        else:
            st.dataframe(drawdowns, width="stretch")

    with report_tabs[3]:
        if trade_dist.empty:
            st.info("暂无完整卖出交易。")
        else:
            st.dataframe(trade_dist, width="stretch")
            chart = trade_dist.set_index("bucket")["count"]
            st.bar_chart(chart, height=260)

    with report_tabs[4]:
        if sector.empty:
            st.info("暂无板块持仓暴露。")
        else:
            st.dataframe(sector, width="stretch")
            chart_cols = [col for col in ["avg_weight", "max_weight"] if col in sector.columns]
            if chart_cols:
                st.bar_chart(sector.set_index("sector")[chart_cols], height=260)


def render_data_check(check_dir: Path) -> None:
    report = read_json_if_exists(check_dir / "data_check_report.json")
    if not report:
        st.info("还没有数据检查报告。点击左上方按钮先运行数据检查。")
        return
    counts = report.get("severity_counts", {})
    cols = st.columns(4)
    cols[0].metric("是否可回测", "是" if report.get("can_backtest") else "否")
    cols[1].metric("错误", counts.get("error", 0))
    cols[2].metric("警告", counts.get("warning", 0))
    cols[3].metric("提示", counts.get("info", 0))
    st.write(
        f"数据范围：{report.get('start_date')} 至 {report.get('end_date')}，"
        f"股票数：{report.get('stock_count')}，行数：{report.get('rows')}"
    )
    issues = pd.DataFrame(report.get("issues", []))
    if issues.empty:
        st.success("未发现数据问题。")
    else:
        st.dataframe(issues.drop(columns=["sample"], errors="ignore"), width="stretch")


def main() -> None:
    st.set_page_config(page_title="A股主升龙头回测", layout="wide")
    st.title("A股主升龙头潜质股回测")
    st.caption("日线潜质识别 + 启动确认 + A股交易约束")

    params = render_sidebar()

    run_cols = st.columns([1, 1, 5])
    with run_cols[0]:
        check_clicked = st.button("数据体检", width="stretch")
    with run_cols[1]:
        backtest_clicked = st.button("运行回测", type="primary", width="stretch")

    if check_clicked:
        with st.spinner("正在检查日线数据..."):
            try:
                report = run_data_check(params["data_path"], params["check_dir"])
                if report["can_backtest"]:
                    st.success("数据检查完成，可以回测。")
                else:
                    st.error("数据检查发现 error，建议先修数据。")
            except Exception as exc:
                st.exception(exc)

    if backtest_clicked:
        with st.spinner("正在运行回测..."):
            try:
                metrics = run_research_backtest(
                    params["data_path"],
                    params["config_path"],
                    params["out_dir"],
                    params["mode"],
                    params["start"],
                    params["end"],
                    params["include_labels"],
                    params["save_features"],
                )
                st.success("回测完成。")
                render_metric_overview(metrics)
            except Exception as exc:
                st.exception(exc)

    tabs = st.tabs(["绩效总览", "绩效报告", "每日预警", "参数稳定性", "净值曲线", "交易明细", "候选与信号", "数据体检报告"])

    with tabs[0]:
        metrics = read_json_if_exists(params["out_dir"] / "metrics.json")
        if metrics:
            render_metric_overview(metrics)
            with st.expander("完整 metrics.json"):
                st.json(metrics)
        else:
            st.info("还没有绩效文件。先运行一次回测。")

    with tabs[1]:
        render_performance_reports(params["out_dir"])

    with tabs[2]:
        render_alerts(params["out_dir"])

    with tabs[3]:
        render_parameter_sweep(params)

    with tabs[4]:
        render_equity(params["out_dir"])

    with tabs[5]:
        render_trades(params["out_dir"])

    with tabs[6]:
        render_signals(params["out_dir"])

    with tabs[7]:
        render_data_check(params["check_dir"])


if __name__ == "__main__":
    main()
