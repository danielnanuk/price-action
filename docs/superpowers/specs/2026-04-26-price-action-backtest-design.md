# Price Action 回测系统设计

**日期:** 2026-04-26
**作者:** danielnanuk@gmail.com (与 Claude 协作)
**状态:** Draft — 待用户审阅

---

## 1. 概述 (Overview)

构建一个基于 Al Brooks《Trading Price Action》体系的**美股 daily swing 回测系统**,用于:

1. **研究** — 量化验证 Brooks 描述的 5 个核心 setup 在 S&P 500 上是否存在统计 edge
2. **手动交易辅助** — 通过标注 K 线图廊"训练眼睛",并提供每个 setup 的期望值数据作为实战参考

本系统**不是**自动化交易系统,**不**生成实时信号,**不**做撮合执行。输出是离线 HTML 研究报告。

## 2. 范围 (Scope)

### 2.1 已锁定的决策

| 维度 | 决策 |
|---|---|
| 标的市场 | 美股 |
| Universe | S&P 500(成员快照取回测起始日) |
| 历史长度 | 5 年(2021-04 ~ 2026-04) |
| 时间框架 | Daily(MVP);后续可扩展 4H/1H |
| 持仓风格 | Swing(数日 ~ 数周) |
| 用户时区 | UTC+8(因此排除日内策略) |
| 实现方案 | Lean Research Pipeline(自写 detector + parquet 缓存 + 静态 HTML 报告)|

### 2.2 MVP 测试的 5 个 setup

| # | Setup | 类型 | 适用 regime |
|---|---|---|---|
| 1 | **H2** (High 2 / 二次回调买入) | 顺势 | bull_trend |
| 2 | **L2** (Low 2 / 二次反弹卖空) | 顺势 | bear_trend |
| 3 | **Bull/Bear Flag Breakout** | 顺势 | trending |
| 4 | **Failed Breakout** (突破失败反转) | 反转 | trading_range |
| 5 | **Double Top / Double Bottom** | 反转 | 不限 |

每个 setup 提供 **strict / standard / loose** 三档参数,各自统计,以观察"严格度 vs 胜率/频率"权衡曲线,防止过拟合到单一阈值。

### 2.3 明确不在 MVP 范围内

- 多 TF 联动分析(daily + 1H/5min)
- 主观 "Always In Long/Short" 判断
- "Major Trend Reversal" 整体格局识别
- Wedge / Climactic / TCL overshoot / Final flag 等 🟡 档 setup
- Scale-out 多档止盈
- 实时信号生成
- Web dashboard / 交互调参

以上均为后续迭代候选。

## 3. 架构 (Architecture)

### 3.1 核心理念

**线性管道 + 中间产物缓存。** 每个阶段读上游 parquet、产出新 parquet,失败可从任意阶段重跑。研究迭代速度取决于此。

```
fetch → indicate → regime → detect → backtest → report
  ↓        ↓         ↓        ↓          ↓          ↓
ohlcv/  indicators/ regime/ candidates/ trades/  reports/
```

### 3.2 关键设计决策

1. **磁盘缓存制 (parquet)** — 改 detector 不必重拉数据,改报告不必重跑 detector
2. **以 ticker 为粒度并行** — `concurrent.futures` 跑满 CPU
3. **CLI 单入口** — `pa-backtest <subcommand>`,子命令:`fetch | indicate | regime | detect | backtest | report | all`
4. **配置驱动** — universe / 日期 / setup 启用列表 / 参数档 / 报告模板均从 `configs/*.yaml` 读
5. **无数据库** — parquet 文件即数据库

### 3.3 目录结构

```
price-action/
├── src/pa/
│   ├── data/         # Massive API 客户端 + parquet 缓存
│   ├── indicators/   # EMA / ATR / swing pivot / bar anatomy
│   ├── regime/       # 趋势分类器 + signal bar 评分器
│   ├── detectors/    # 5 个 setup 检测器
│   ├── backtest/     # 入场/止损/止盈模拟
│   ├── report/       # HTML + 标注图
│   └── cli.py
├── data/             # parquet 缓存(gitignored)
├── reports/          # HTML 输出(gitignored)
├── configs/          # YAML 配置
├── tests/            # pytest
└── pyproject.toml
```

### 3.4 工具链

- Python 3.11+
- 依赖管理:`uv`
- 类型检查:`mypy --strict`
- Lint:`ruff`
- 测试:`pytest` + `hypothesis`(property-based)
- 不引入 `talib`(避免 C 依赖,纯 numpy/pandas 向量化)
- 不引入 backtest 框架(Brooks 多 K 线形态自写更顺)

## 4. 组件 (Components)

每个组件职责单一、纯函数化(输入 DataFrame → 输出 DataFrame),无状态。

### 4.1 `pa.data` — 数据获取 + 缓存
- 调用 Massive API(Polygon-compatible)拉取 OHLCV
- 一只股票一个 parquet 文件,`(ticker, start, end)` 哈希命名;命中即跳过
- 仅使用 split/dividend adjusted 数据
- 接口:`fetch_ohlcv(ticker: str, start: date, end: date) -> DataFrame`

### 4.2 `pa.indicators` — 派生指标
- `ema(close, n)` — n=20, 50
- `atr(high, low, close, n=14)` — 用于止损/动量阈值
- `swing_pivot(high, low, n=2)` — Brooks 风格 swing high/low(N 根 K 线两侧最高/低)
- `bar_anatomy(df)` — 每根 K 线的元数据:body%、上影%、下影%、是否 bull、相对前一根关系
- 纯 numpy 向量化,无 talib

### 4.3 `pa.regime` — 趋势分类 + signal bar 评分
- `classify_regime(df) -> Series[regime_label]` — 标签 ∈ {bull_trend, bear_trend, trading_range, transitional}
- `signal_bar_score(df) -> Series[float]` — 0..1 综合评分(实体大小 / 上下影 / 收盘位置)

**Regime 规则(standard 档,strict/loose 仅在阈值上调整):**
- `bull_trend`:close > EMA20 > EMA50,且最近 20 根 K 线至少一次 higher high + higher low,且 EMA20 上升斜率 > 阈值
- `bear_trend`:对称
- `trading_range`:既非 bull 亦非 bear,且 close 在 EMA20 ± 1.5 ATR 内振荡
- `transitional`:边缘情况(用于过滤,不作为任何 setup 的有效 regime)

### 4.4 `pa.detectors` — Setup 检测器

统一接口:
```python
def detect(bars: BarsFrame, params: SetupParams) -> CandidatesFrame:
    """
    bars:   单一 ticker 的 OHLCV + indicators + regime 合并表。
            必须含: date, open, high, low, close, volume, ema20, ema50, atr14,
                    swing_high, swing_low, bar_body_pct, bar_upper_wick_pct,
                    bar_lower_wick_pct, bar_is_bull, bar_close_position,
                    regime, regime_strength, signal_bar_score
    params: SetupParams 实例(tier ∈ {strict, standard, loose} + 该 setup 的阈值字典)
    返回:   candidates 表(§5.1 Stage 4 schema)
    """
```

每个 setup 一个文件,详细规则见 §6。

### 4.5 `pa.backtest` — 入场/止损/止盈模拟
- 接口:`simulate(candidates: DataFrame, ohlcv: DataFrame) -> DataFrame[trades]`
- 引擎是**通用执行器**:`candidates` 表的每一行已携带 `entry_price`、`stop_price`、`target_price`,引擎只负责按 OHLCV 序列模拟从 signal_date 起的入场/止损/止盈/时间止损流程,**不**做策略层判断。各 setup 的具体止损/止盈逻辑由 detector 在 §6 中规定并写入 candidates 表。
- **统一时间止损:** 20 个交易日未触发任何止损/止盈即按收盘价平仓(适用所有 setup)
- **不做** scale-out(MVP 简化)
- **同根 K 线模糊性处理:** 止损止盈在同一根 K 线触发时,**保守假设先止损**,在 trade 表标 `same_bar_ambiguous=true`

### 4.6 `pa.report` — HTML 报告渲染
- `reports/<run-id>/index.html` — 总览,每 setup × 每参数档的统计表
- `reports/<run-id>/<setup>.html` — 单 setup 详情 + 样本图廊
- `reports/<run-id>/samples/<setup>/<ticker>_<date>.png` — 单样本标注图(K 线 + 入场/止损/止盈线 + setup 形态边框)
- `reports/<run-id>/data.parquet` — 平铺所有 trade,供 jupyter 二次分析

## 5. 数据流 (Data Flow)

### 5.1 各 stage 的 schema

#### Stage 1 — `fetch`
```
input:  config.yaml (universe, date_range)
output: data/ohlcv/<ticker>.parquet
        cols: [date, open, high, low, close, volume, vwap]
```

#### Stage 2 — `indicate`
```
input:  data/ohlcv/<ticker>.parquet
output: data/indicators/<ticker>.parquet
        cols: [date, ema20, ema50, atr14, swing_high, swing_low,
               bar_body_pct, bar_upper_wick_pct, bar_lower_wick_pct,
               bar_is_bull, bar_close_position]
```

#### Stage 3 — `regime`
```
input:  ohlcv + indicators
output: data/regime/<ticker>.parquet
        cols: [date, regime, regime_strength, signal_bar_score]
```

#### Stage 4 — `detect`
```
input:  ohlcv + indicators + regime + config
output: data/candidates/<setup>_<param_tier>.parquet  (跨 ticker 合并)
        cols: [ticker, signal_date, side, entry_price, stop_price, target_price,
               setup_score, regime_at_signal, params_tier]
```
注:candidates 是"形态出现"层,**未模拟入场**。

#### Stage 5 — `backtest`
```
input:  candidates + ohlcv
output: data/trades/<setup>_<param_tier>.parquet
        cols: [ticker, signal_date, entry_date, entry_price, exit_date, exit_price,
               exit_reason, pnl_r, pnl_pct, mae_r, mfe_r, days_held,
               regime_at_signal, same_bar_ambiguous]
```
- `pnl_r` — 以 R(初始风险单位)度量
- `mae` / `mfe` — Max Adverse / Favorable Excursion(用于训练眼睛)
- `exit_reason` ∈ {target_hit, stop_hit, time_stop, end_of_data}

#### Stage 6 — `report`
- index.html、单 setup 详情页、样本图廊、可二次分析的 data.parquet

### 5.2 报告必含统计

每 setup × 每参数档:

| 指标 | 说明 |
|---|---|
| N | 样本总数 |
| Win Rate | exit=target 的比例 |
| 平均 R | 整体期望值 |
| Profit Factor | 总盈利 / 总亏损 |
| 平均 MAE / MFE | 浮亏/浮盈极值 |
| 按 regime 分层 | bull / bear / range 三栏 |
| 按年分层 | 看 edge 是否随时间稳定 |
| 严格度对比 | strict vs standard vs loose 的 win rate × frequency |

## 6. Setup 详细定义

> 以下规则为 **standard 档**。`strict` 收紧阈值(如更高 signal bar 评分、更窄回调形状);`loose` 放宽。**强约束:** loose 候选集合 ⊇ standard ⊇ strict(单调性,详见 §8 测试)。

### 6.1 H2 — High 2(二次回调买入)

**前置:** `regime == bull_trend`

**形态识别:**
- 先有一段上升:近 20 根 K 线创出 swing high `H_top`
- 第一腿回调:从 `H_top` 起,至少 2 根 K 线收盘下行,形成 swing low `L1`
- 反弹:至少 2 根 K 线收盘上行(不要求创新高)
- 第二腿回调:再次至少 2 根 K 线收盘下行,形成 swing low `L2`(`L2` 不要求严格高于 `L1`,允许小幅穿透)
- Signal bar:`L2` 之后第一根 bull bar(收盘高于开盘),且 `signal_bar_score >= 0.5`

**入场:** 突破 signal bar 高点 1 tick 时市价买入

**止损:** signal bar 低点 - 0.5 ATR

**止盈:** 入场价 + 2R

### 6.2 L2 — Low 2(二次反弹卖空)

H2 的对称镜像,前置 `regime == bear_trend`,signal bar 为 bear bar。

### 6.3 Bull/Bear Flag Breakout(旗形突破)

**前置:** `regime ∈ {bull_trend, bear_trend}`

**形态识别(Bull Flag,Bear 对称):**
- 强势冲击腿:连续 N=3+ 根 same-direction bar,或者 ATR 显著扩张(冲击腿幅度 > 1.5 × ATR(20))
- 紧凑整理:随后 5+ 根 K 线,整体波动范围 < 冲击腿幅度 × 0.5
- 突破确认:某根 K 线 close > 整理高点

**入场:** 突破 K 线收盘后,下一根开盘市价买入

**止损:** 整理低点 - 0.5 ATR

**止盈:** 冲击腿幅度从突破点投影(measured move),上限 3R

### 6.4 Failed Breakout(突破失败反转)

**前置:** `regime == trading_range`

**形态识别(以失败的向上突破→做空为例;向下失败做多对称):**
- 识别近 20 根 K 线的震荡区间 `[range_low, range_high]`
- 突破 K 线:某根 K 线 close > `range_high`
- 失败确认:接下来 K=3 根 K 线内,某根 K 线 close < `range_high`(回到区间内)

**入场:** 失败确认 K 线收盘后,下一根开盘市价做空

**止损:** 突破 K 线高点 + 0.5 ATR

**止盈:** 取 2R 目标(`入场价 - 2R`)与反向区间边界(`range_low`)中**离入场价更近**的那个,作为目标价。即对做空场景:`target = max(入场价 - 2R, range_low)`(取较高值 = 较保守、较快达成)。做多场景对称使用 `min(入场价 + 2R, range_high)`。

### 6.5 Double Top / Double Bottom(双顶/双底)

**前置:** 任意 regime(反转 setup)

**形态识别(Double Top,Double Bottom 对称):**
- 第一个峰 `P1` = 近 30 根 K 线的 swing high
- 回调:从 `P1` 至少 5 根 K 线下行,形成中间低点 `M`,`M < P1 - 1 × ATR`
- 第二个峰 `P2`:回升后再创高点,`|P2 - P1| < 5% × P1`(strict: 2%, loose: 8%)
- Signal bar:`P2` 之后第一根 bear bar,`signal_bar_score >= 0.5`

**入场:** signal bar 低点突破 1 tick 时做空

**止损:** max(P1, P2) + 0.5 ATR

**止盈:** 入场价 - 2R

## 7. 错误处理

### 7.1 失败处理矩阵

| 故障类型 | 处理 |
|---|---|
| API 网络 / 5xx / 超时 | 指数退避重试 3 次 → 仍失败则跳过 ticker,记 `data/_failures/fetch.jsonl` |
| API 限流 (429) | token bucket 整体降速,polygon skill 已处理 |
| Ticker 该期内无数据 | 0 行 + warning,继续 |
| 数据缺口 | parquet 标 `has_gaps=true`,detector 跳过缺口处形态 |
| 拆股/分红 | 仅用 adjusted 数据,不二次复权 |
| 样本数过少(N < 5) | 报告标 `low confidence` |
| 配置错误 | 启动 schema 校验,fail-fast 抛 `ConfigError` |
| 指标 NaN(K 线不足) | 头部 NaN 丢弃,detector `.dropna()` |
| 同根 K 线止盈止损同发 | 保守先止损,标 `same_bar_ambiguous=true` |
| 形态 0 匹配 | 不报错,空表,报告显示 0 样本 |

### 7.2 结构化日志

每阶段写 JSON line 到 `data/_logs/<stage>_<run-id>.jsonl`:
```json
{"ts": "...", "level": "warn", "stage": "fetch", "ticker": "AAPL", "msg": "...", "duration_ms": 234}
```

### 7.3 数据真实性自检

每次 fetch 后自动跑:
- 单根 K 线涨跌幅 > 30%(可能拆股未识别)
- 连续 N 天成交量为 0
- 实际数据长度 < 配置 80%(IPO/退市/缺数据严重)— 仍纳入但打 `coverage` 标签

### 7.4 明确不做

- 自动数据修复(只检测警告,不静默改写)
- 自动重试失败 ticker(留给用户手动重跑)
- 异常监控告警(单机研究项目,无 Sentry)

## 8. 测试策略

### 8.1 测试金字塔

```
Visual Regression (~5 个 baseline PNG)
E2E Pipeline (3 ticker × 1 年 < 30s)
Detector Integration (每 setup 10 个 fixture)
Property / Invariant (hypothesis 生成)
Unit (indicators/math) — 最广,确定性
```

### 8.2 关键测试

**Unit:** EMA / ATR / swing pivot 用已知输入手算对比。退出模拟边界:同根触发、跨缺口、时间止损边界。

**Property-based(hypothesis):**
- Trade ledger 不变量:`entry_date ≤ exit_date`、`exit_price ∈ [bar_low, bar_high]`、`pnl_r` 有限
- Regime 全覆盖:任意输入每根 K 线都有 regime 标签(无 NaN)
- **Detector 单调性:** loose ⊇ standard ⊇ strict (强约束,见下节)
- 确定性:同输入两次跑结果完全相同

**Detector Integration(命脉):**
每 setup `tests/fixtures/<setup>/` 下 10 个手挑案例(5 positive + 5 negative),每个 fixture 是一段 OHLCV + expected detection。每次 detector 调参立刻看到回归。

**E2E:** 迷你 universe 端到端 < 30 秒,断言所有 stage parquet 存在且 schema 正确,HTML 渲染成功。

**Visual Regression:** 5 个教科书 fixture 渲染标注图,与 baseline PNG 像素差 < 1%。

### 8.3 单调性约束的设计含义

`loose ⊇ standard ⊇ strict` 意味着:**严格度只能调阈值,不能换判法。** 例如 H2 的 strict 只能要求"signal bar score 更高"或"两腿回调更深",而不能换成"加一个 EMA 过滤"——后者不构成 superset 关系。这保证了"严格度"在三档之间有可解释的语义连续性。

### 8.4 基础设施

- Runner:`pytest -n auto`(并行)
- 覆盖率:`pytest --cov=pa --cov-report=term-missing`,目标 ≥ 85%(标注图渲染豁免)
- 预提交:`pre-commit` 跑 ruff / mypy
- **不做 GitHub Actions CI**(本地够用)

## 9. 开放问题与后续工作

### 9.1 MVP 完成后的扩展候选

按优先级:

1. **🟡 档 setup**(Wedge / Outside bar / Climactic / TCL overshoot / Final flag)
2. **多 TF 联动**(daily 找 setup + 1H 精确入场)
3. **Trailing stop / scale-out** 退出策略
4. **Walk-forward 优化**(现在是固定 5 年回测,未来可加滚动窗口验证)
5. **交互式 Streamlit dashboard**(从静态报告升级)
6. **跨样本 normalize 叠加图**(同 setup 的所有样本路径标准化后叠加,看典型形态)

### 9.2 风险与已知限制

- **5 年历史 regime 不平衡:** 2022 是唯一真熊市,bear-side setup 样本会偏少。报告按 regime 分层呈现,如统计 power 不够再扩到 10 年
- **S&P 500 成员变动:** 用回测起始日的成员快照(避免 survivorship bias 部分,但小盘退出样本仍缺)。后续可改为用历史成员动态名单
- **Daily 数据无盘前盘后:** 不影响 swing,但 gap up/down 会影响入场假设。我们用次日开盘价模拟入场以贴近实盘
- **Massive API 数据质量未独立验证:** 如发现可疑值,回归到 Polygon/Yahoo 交叉验证

## 10. 验收标准

MVP 完成的判断标准:

- [ ] 全部 6 个 stage 端到端跑通(`pa-backtest all` 一键执行)
- [ ] S&P 500 × 5 年数据完整缓存到 parquet
- [ ] 5 个 setup × 3 参数档 = 15 组结果均产出 candidates / trades 表
- [ ] HTML 报告渲染完整,含统计表 + 标注图 + regime 分层 + 年度分层
- [ ] 测试金字塔覆盖:`pytest` 全绿,覆盖率 ≥ 85%
- [ ] 每 setup 10 fixture 全通过
- [ ] 单调性约束 (loose ⊇ standard ⊇ strict) 在 hypothesis 测试下成立
- [ ] 全 pipeline 在单机(8 核 16GB)上 30 分钟内完成
