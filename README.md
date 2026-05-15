# A股主升龙头潜质股量化回测程序

这是一个基于日线数据的 A 股“主升龙头潜质股”回测 MVP。程序按你给出的研究框架实现成两层：

1. 潜质识别：收盘后计算板块边际改善、个股隐性强度、抗跌性、波动率收缩、温和吸筹、临界位置等因子，得到潜质分。
2. 启动确认：候选股只有出现突破、放量、强势日、板块共振、收盘位置较强等触发条件后，才进入交易。

回测会按 A 股交易约束执行，包括：

- t 日收盘生成信号，t+1 日开盘成交
- 涨停开盘或一字涨停不可买
- 跌停开盘或一字跌停不可卖
- 停牌不可交易
- T+1，当日买入不能当日卖出
- 佣金、过户费、印花税、滑点、成交容量限制

## 有没有界面？

有。本项目提供一个本地 Streamlit Web 界面：

第一次使用前先安装依赖：

```powershell
pip install -r requirements.txt
```

```powershell
.\scripts\run_dashboard.ps1
```

默认打开：

```text
http://127.0.0.1:8501
```

如果端口被占用，可以指定端口：

```powershell
.\scripts\run_dashboard.ps1 -Port 8502
```

界面可以完成：

```text
选择日线数据路径
选择配置文件和回测模式
运行数据体检
运行回测
查看绩效指标
查看分年/月度绩效报告
查看净值曲线和回撤曲线
查看交易明细
查看每日候选池、触发信号和买入信号
查看每日预警表并下载最新预警
运行参数稳定性测试
查看数据体检报告
```

你也可以继续用命令行运行。

运行后会生成 CSV 和 JSON 结果文件，包括：

```text
equity.csv      每日权益曲线
trades.csv      交易明细
positions.csv   每日持仓
signals.csv     候选池、触发信号和买入信号，默认只保留相关股票
alerts.csv      每日预警表
alerts_latest.csv 最新交易日预警表
metrics.json    总体绩效和命中率指标
yearly_performance.csv 年度绩效
monthly_returns.csv    月度收益矩阵
drawdown_periods.csv   主要回撤区间
trade_distribution.csv 交易收益分布
sector_exposure.csv    板块持仓暴露
```

后续如果需要，可以再加一个 Streamlit / Dash 图形界面，用来查看净值曲线、交易明细、TopK 候选股和每日预警表。

## 快速运行样例

项目自带一个合成样例数据生成器，用来验证程序链路，不代表真实策略收益。

在当前目录运行：

```powershell
python main.py generate-sample --out data/raw
python main.py check-data --data data/raw/daily_price.csv --out data/data_check
python main.py backtest --config config/default.toml --data data/raw/daily_price.csv --out data/backtest_result/confirmed --mode confirmed
```

如果 Windows 的 `python` 命令不可用，可以尝试：

```powershell
py -3 main.py generate-sample --out data/raw
py -3 main.py check-data --data data/raw/daily_price.csv --out data/data_check
py -3 main.py backtest --config config/default.toml --data data/raw/daily_price.csv --out data/backtest_result/confirmed --mode confirmed
```

也可以直接运行脚本：

```powershell
.\scripts\run_sample_confirmed.ps1
```

启动可视化界面：

```powershell
.\scripts\run_dashboard.ps1
```

## 三种回测模式

`confirmed`

推荐主模式。先进入潜质池，再等启动确认信号，次日开盘买入。

```powershell
python main.py backtest --config config/default.toml --data data/raw/daily_price.csv --out data/backtest_result/confirmed --mode confirmed
```

`potential`

基准模式。只按潜质分排名买入，不等待启动确认。用于检查因子本身有没有预测力。

```powershell
python main.py backtest --config config/default.toml --data data/raw/daily_price.csv --out data/backtest_result/potential --mode potential
```

`hybrid`

混合模式。潜质分进入前列先买半仓，出现启动确认后加到目标仓位。

```powershell
python main.py backtest --config config/default.toml --data data/raw/daily_price.csv --out data/backtest_result/hybrid --mode hybrid
```

## 真实 A 股回测需要你提供什么数据？

最低需要一张点时化日线表，可以是 CSV 或 Parquet。默认路径是：

```text
data/raw/daily_price.csv
```

必需字段如下：

```text
date        交易日期
code        股票代码
open        开盘价
high        最高价
low         最低价
close       收盘价
pre_close   前收盘价
volume      成交量
amount      成交额
up_limit    当日涨停价
down_limit  当日跌停价
paused      是否停牌，true/false
is_st       当日是否 ST 或风险警示，true/false
list_date   上市日期
sector      当日所属行业或板块
float_mv    流通市值
total_mv    总市值
turnover    换手率
```

推荐字段如下。没有这些字段也能运行，但预警表会不够直观：

```text
ts_code     带交易所后缀的股票代码，例如 000001.SZ
name        股票名称
exchange    交易所，例如 SSE / SZSE / BSE
market      市场类型，例如 主板 / 创业板 / 科创板 / 北交所
```

非常重要：

- `sector` 必须是历史点时化行业或板块，不能用今天的行业/概念成分回填历史。
- `is_st` 必须是历史状态，不能用今天是否 ST 回填历史。
- `paused` 必须是历史停牌状态。
- `up_limit`、`down_limit` 必须是当日真实涨跌停价，要能区分主板、创业板、科创板、新股前五日等情况。
- 价格最好使用前复权或后复权口径，但交易撮合价格、涨跌停价和成交额口径必须一致。

## 数据源可以来自哪里？

程序本身不绑定具体数据供应商。你可以用任意数据源，只要整理成上面的字段即可。

常见来源包括：

- Tushare Pro
- JoinQuant / 聚宽
- RiceQuant / 米筐
- Wind
- Choice
- iFinD
- Qlib 本地 A 股数据
- 自己维护的数据库或 Parquet 文件

如果只是先试程序，可以直接用样例数据：

```powershell
python main.py generate-sample --out data/raw
```

如果要做真实研究，建议至少准备 5 年以上日线数据，并保证 ST、停牌、涨跌停、上市日期、行业归属都是历史点时化数据。

## 第一步：检查日线数据质量

正式回测前建议先运行数据体检：

```powershell
python main.py check-data --data data/raw/daily_price.csv --out data/data_check
```

检查报告会输出到：

```text
data/data_check/
```

包括：

```text
data_check_report.json   完整检查报告
data_check_summary.json  摘要
data_check_issues.csv    异常列表
```

检查内容包括：

```text
必需字段是否齐全
date + code 是否重复
价格字段是否为正
open/close 是否在 high/low 区间内
涨跌停价格是否异常
收盘价是否超过涨跌停价
停牌记录是否仍有成交量/成交额
非停牌记录是否没有成交
上市日期是否缺失或晚于交易日
行业/板块字段是否缺失过多
成交额单位是否疑似错误
流通市值/总市值是否异常
paused/is_st 是否是布尔风格字段
```

如果报告里的 `can_backtest=false`，建议先修数据，不要直接看策略结果。

## 使用 Tushare 数据源

程序已经内置 `fetch-tushare` 命令，可以把 Tushare 数据整理成回测需要的 `daily_price.csv`。

先安装依赖：

```powershell
pip install -r requirements.txt
```

长历史下载建议启用缓存和重试，避免接口中断后从头开始：

```powershell
python main.py fetch-tushare `
  --start 20210101 `
  --end 20260513 `
  --out data/raw/daily_price_long.csv `
  --cache-dir data/raw/tushare_cache/daily_20210101_20260513 `
  --sleep 0.05 `
  --retries 5 `
  --retry-sleep 5
```

`--cache-dir` 会按交易日保存中间文件，后续重复运行时会自动跳过已经下载成功的交易日。

如果长历史数据体检提示少数股票 `list_date` 缺失或晚于首个交易日，可以生成一个清洗副本：

```powershell
python main.py fix-daily `
  --input data/raw/daily_price_long.csv `
  --out data/raw/daily_price_long_clean.csv
```

清洗会保留原文件不动，只修正常见的上市日期兜底和布尔字段口径。

设置 token。建议用环境变量，不要把 token 写进代码文件：

```powershell
$env:TUSHARE_TOKEN="你的token"
$env:TUSHARE_HTTP_URL="http://101.35.233.113:8020/"
```

Tushare 统一初始化方式保存在：

```text
docs/tushare_init.md
dragon_backtest/tushare_client.py
```

拉取数据，例如 2021 年到 2024 年：

```powershell
python main.py fetch-tushare --start 20210101 --end 20241231 --out data/raw/daily_price.csv
```

先检查数据：

```powershell
python main.py check-data --data data/raw/daily_price.csv --out data/data_check
```

然后运行回测：

```powershell
python main.py backtest --config config/default.toml --data data/raw/daily_price.csv --out data/backtest_result/confirmed --mode confirmed
```

或者用脚本一键拉取并回测：

```powershell
.\scripts\fetch_tushare_and_backtest.ps1 -Start 20210101 -End 20241231 -Mode confirmed
```

`fetch-tushare` 会调用这些数据：

```text
trade_cal      交易日历
stock_basic    股票基础信息、上市日期、当前行业
daily          日线行情
daily_basic    换手率、市值
stk_limit      涨跌停价格
namechange     历史名称变化，用于尽量还原 ST 状态
```

注意：`stock_basic` 的行业字段通常不是严格点时化行业成分。第一版可以先跑通；正式研究建议后续接入申万/中信等历史行业成分表，替换 `sector` 字段，避免未来函数。

## 使用 minishare 分钟线数据源

minishare 可以用于第二阶段分钟级研究，例如：

- 9:35 后判断突破
- 用 5 分钟或更高频率的 VWAP 模拟成交
- 验证首板、炸板、回封、尾盘强度
- 改进涨停排队和可成交性判断

当前程序先提供分钟数据下载器。主回测器仍是日线版本，分钟撮合逻辑可以在数据稳定后继续扩展。

安装：

```powershell
pip install minishare --upgrade
```

设置授权码。建议使用环境变量，不要把授权码写进代码文件：

```powershell
$env:MINISHARE_TOKEN="你的授权码"
```

拉取单只股票 5 分钟数据：

```powershell
python main.py fetch-minishare-mins --codes 600000.SH --freq 5min --start "20250210 09:00:00" --end "20250210 19:00:00"
```

拉取多只股票：

```powershell
python main.py fetch-minishare-mins --codes 600000.SH,000001.SZ --freq 5min --start "20250210 09:00:00" --end "20250210 19:00:00" --combine
```

也可以把股票列表放到文件里：

```powershell
python main.py fetch-minishare-mins --codes-file data/raw/minute_codes.csv --freq 15min --start "20250210 09:00:00" --end "20250210 19:00:00"
```

输出路径：

```text
data/raw/minute/5min/
```

支持频率：

```text
5min / 15min / 30min / 60min
```

注意：你提供的 minishare 分钟数据起始时间是 2025-01-01，因此分钟级回测只能从 2025 年以后开始。日线回测仍可使用 Tushare 或其他日线数据源做更长历史。

## 配置参数在哪里改？

主要参数在：

```text
config/default.toml
```

常用参数：

```text
[universe]
min_avg_amount_20d        20日日均成交额下限
max_ret_20d               20日涨幅上限
max_limit_up_count_10d    近10日涨停次数上限
min_close_to_high_60d     距离60日高点的最低比例

[signal]
mode                      confirmed / potential / hybrid
candidate_top_n           每日候选池数量
trigger_amount_to_ma60    启动确认的放量阈值
trigger_ret_1d            启动确认的当日涨幅阈值

[trade]
initial_cash              初始资金
target_weight             单票目标仓位
max_positions             最大持仓数
commission_rate           佣金
buy_slippage              买入滑点
sell_slippage             卖出滑点
volume_cap                单笔订单占当日成交额上限

[risk]
stop_loss                 止损
trailing_stop             移动止盈回撤
max_holding_days          最长持仓天数
stale_exit_days           买入后迟迟不启动的退出天数
```

## 输出结果怎么看？

回测完成后查看：

```text
data/backtest_result/confirmed/metrics.json
```

里面包括：

```text
total_return              总收益
annual_return             年化收益
max_drawdown              最大回撤
sharpe                    夏普比率
calmar                    Calmar 比率
win_rate                  胜率
avg_trade_return          单笔平均收益
tradable_buy_rate         买入信号可成交比例
limit_buy_block_count     涨停不可买次数
limit_sell_block_count    跌停不可卖次数
top10_precision_start_10d Top10 未来10日启动命中率
top10_leader_hit_rate_20d Top10 未来20日龙头命中率
top10_avg_lead_days       候选到启动的平均提前天数
```

交易明细看：

```text
data/backtest_result/confirmed/trades.csv
```

每日候选和信号看：

```text
data/backtest_result/confirmed/signals.csv
```

为避免真实全市场回测输出过大，`signals.csv` 默认只保存候选、触发和买入相关记录，不保存全市场所有股票的完整因子。需要完整特征明细时，可以在回测命令后追加：

```powershell
--save-features
```

## 每日预警表

回测完成后会自动生成：

```text
data/backtest_result/confirmed/alerts.csv
data/backtest_result/confirmed/alerts_latest.csv
```

预警表字段包括：

```text
date                  日期
code                  股票代码
ts_code               带交易所后缀的股票代码
name                  股票名称
exchange              交易所
market                市场类型
sector                板块/行业
action                建议动作
reason                触发原因
score                 潜质分
candidate_rank        候选排名
sector_shift_pct      板块边际改善
hidden_rs_score       个股隐性强度
accumulation_score    温和吸筹
vol_squeeze_score     波动率收缩
anti_fall_score       抗跌性
position_score        临界位置
volatility_energy_score 波动能量
trigger               是否启动确认
buy_signal            是否买入信号
```

建议动作当前包括：

```text
次日可买    潜质池 + 启动确认
观察        潜质分靠前，但尚未触发
过热风险    成交额短期放大过高
转弱剔除    板块边际强度转弱
```

也可以单独从已有信号表生成预警：

```powershell
python main.py build-alerts --signals data/backtest_result/confirmed/signals.csv --out data/backtest_result/confirmed --top-n 30
```

## 参数稳定性测试

不要只看一组参数。可以用 `sweep-params` 批量测试候选数量、启动放量阈值、止损、最长持仓天数。

示例：

```powershell
python main.py sweep-params `
  --config config/default.toml `
  --data data/raw/daily_price.csv `
  --out data/param_sweep `
  --mode confirmed `
  --candidate-top-n 30,50 `
  --trigger-volume 1.3,1.5 `
  --stop-loss -0.08,-0.10 `
  --max-holding-days 20,30
```

输出：

```text
data/param_sweep/sweep_results.csv
data/param_sweep/sweep_top20.csv
```

重点观察：

```text
total_return         总收益
max_drawdown         最大回撤
sharpe               夏普
round_trip_count     完整交易笔数
tradable_buy_rate    信号可成交比例
top10_precision_start_10d  Top10未来10日启动命中率
return_to_drawdown   收益回撤比
```

如果只有某一组参数有效，而邻近参数都失效，大概率是过拟合。更健康的情况是：参数轻微变化后，收益、回撤、命中率结构仍然相对稳定。

## 分年/月度绩效报告

回测完成后会自动生成：

```text
yearly_performance.csv
monthly_returns.csv
drawdown_periods.csv
trade_distribution.csv
sector_exposure.csv
```

这些报告用于回答：

```text
策略主要在哪些年份赚钱？
月度收益是否只靠极少数月份？
最大回撤从哪天开始、哪天触底、是否修复？
单笔交易收益分布是否健康？
收益或持仓是否过度集中在某个板块？
```

如果你已经有回测输出，也可以不重新回测，单独生成报告：

```powershell
python main.py build-reports --input data/backtest_result/confirmed
```

如果回测已经写出 `equity.csv/trades.csv/signals.csv`，但中途没来得及生成 `metrics.json`、预警和报告，可以补生成：

```powershell
python main.py summarize-backtest `
  --input data/backtest_result/confirmed `
  --config config/default.toml
```

## 时间切片与市场环境归因

为了检查收益是否集中在少数阶段，可以运行市场环境归因：

```powershell
python main.py analyze-regimes `
  --data data/raw/daily_price.csv `
  --input data/backtest_result/event_tuned
```

输出文件：

```text
market_context.csv             全市场等权收益、成交额、赚钱效应、涨停数量等环境指标
time_slice_performance.csv     年度、半年度、季度、月度收益切片
market_regime_performance.csv  按趋势、流动性、赚钱效应拆分的策略表现
monthly_contribution.csv       月度贡献和当月主导市场环境
regime_summary.json            摘要：最好/最差月份和市场环境
```

市场环境由全市场日线自动合成，不依赖额外指数数据：

```text
trend_regime       uptrend / range / downtrend
liquidity_regime   expanding / neutral / shrinking
breadth_regime     strong_breadth / neutral_breadth / weak_breadth
combined_regime    趋势 + 流动性组合
```

归因后可以测试市场环境过滤版配置：

```powershell
python main.py backtest `
  --config config/event_tuned_market_filter.toml `
  --data data/raw/daily_price.csv `
  --out data/backtest_result/event_tuned_market_filter `
  --mode confirmed
```

这版会在买入前过滤：

```text
弱赚钱效应：market_breadth_regime = weak_breadth
震荡中性成交：market_trend_regime = range 且 market_liquidity_regime = neutral
近5日上涨家数占比低于 45%
```

也可以测试更强的市场环境退出版：

```powershell
python main.py backtest `
  --config config/event_tuned_market_exit.toml `
  --data data/raw/daily_price.csv `
  --out data/backtest_result/event_tuned_market_exit `
  --mode confirmed
```

这版不仅过滤新买入，持仓期间如果市场环境失效，也会触发 `market_filter_exit` 卖出。

当前真实数据测试中，市场过滤/退出版暂时不作为主线配置：`event_tuned_market_filter` 降低了收益且回撤变大，`event_tuned_market_exit` 卖出过于频繁。主线仍是 `event_tuned.toml`，市场环境更适合下一步做动态仓位，而不是直接过滤或清仓。

动态仓位实验配置：

```powershell
python main.py backtest `
  --config config/event_tuned_dynamic_sizing.toml `
  --data data/raw/daily_price.csv `
  --out data/backtest_result/event_tuned_dynamic_sizing `
  --mode confirmed
```

这版不删除信号，只根据市场环境调整新开仓的 `target_weight_mult`：

```text
强赚钱效应：1.0
普通赚钱效应：0.8
弱赚钱效应：0.5
上升趋势：1.0
震荡：0.8
下跌趋势：0.6
震荡 + 中性成交：最高 0.6
```

轻量动态仓位配置：

```powershell
python main.py backtest `
  --config config/event_tuned_dynamic_sizing_light.toml `
  --data data/raw/daily_price.csv `
  --out data/backtest_result/event_tuned_dynamic_sizing_light `
  --mode confirmed
```

当前测试中，轻量版表现更均衡：收益低于主线，但最大回撤也更低。主线和稳健备选如下：

```text
收益优先：config/event_tuned.toml
稳健备选：config/event_tuned_dynamic_sizing_light.toml
```

## 爆发初期事件研究

在继续调策略前，建议先做事件研究：从真实数据里找到“触发启动”的样本，再按未来 20 日表现分成成功启动、失败启动和灰样本，反推启动前特征。

运行：

```powershell
python main.py study-start-events `
  --config config/default.toml `
  --data data/raw/daily_price.csv `
  --out data/event_study
```

默认事件定义：

```text
触发事件：
收盘突破60日高点
成交额MA5/MA60 > 1.5
当日涨幅 > 5% 或涨停
板块边际强度分位 >= 70%

成功启动：
触发后20日最大涨幅 >= 25%
且未来20日板块内排名前10%

失败启动：
触发后20日最大涨幅 <= 10%
且未来20日板块内排名后70%
```

输出：

```text
data/event_study/start_events.csv
data/event_study/pre_start_feature_profile.csv
data/event_study/positive_vs_negative.csv
data/event_study/feature_effects.csv
data/event_study/feature_quantiles.csv
data/event_study/event_samples.csv
data/event_study/event_summary.csv
data/event_study/event_rate_by_market.csv
data/event_study/event_rate_by_sector.csv
```

这些表用于回答：

```text
真正成功启动的股票，启动前5/10/20天有什么共同特征？
失败启动和成功启动在放量、涨幅、板块强度、隐性强度上的差异是什么？
哪些阈值应该收紧？
哪些指标权重应该提高或降低？
```

## 事件研究调优版配置

项目保留默认配置 `config/default.toml`，并新增一版根据事件研究结果调整的配置：

```text
config/event_tuned.toml
```

这版配置的核心变化是：

```text
提高：个股隐性相对强度、板块边际改善、临界位置
新增：波动能量因子 volatility_energy_score
降低：低波动收缩、抗跌性
放宽：近20日涨幅、近10日涨停次数、成交吸筹阈值
收窄：候选池从 Top50 调整为 Top40
```

运行方式：

```powershell
python main.py backtest `
  --config config/event_tuned.toml `
  --data data/raw/daily_price.csv `
  --out data/backtest_result/event_tuned `
  --mode confirmed
```

调优版仍然只是研究配置，不代表已经找到稳定可实盘的策略。它的作用是把“成功启动样本的统计特征”先落进回测器，再通过样本外、参数稳定性和交易约束继续筛掉偶然性。

当前参数稳定性测试中，`event_tuned` 附近 36 组参数有 27 组总收益为正，其中 `trigger_amount_to_ma60=1.5` 的 12 组全部为正；`1.3` 过宽，失败率较高。因此调优版保留 1.5 放量阈值，并把候选池设为 Top40。

## 当前版本边界

当前是日线研究版，适合先验证：

- 因子有没有预测力
- 潜质池是否能提前发现启动股
- 启动确认是否能减少假信号
- A 股交易约束后收益是否还存在

当前没有实现：

- 分钟线撮合
- 集合竞价数据
- Level-2 数据
- 龙虎榜、公告、新闻等事件数据
- LightGBM / XGBoost 机器学习模型
- 图形化回测界面

建议先用真实日线数据跑通，再扩展机器学习模型和分钟线版本。

