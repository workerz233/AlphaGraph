# AlphaGraph

AlphaGraph 是一个面向研报驱动股票图谱建模的研究管线：从行情、财务和研报 Markdown 构建股票图与知识图谱，训练 GNN 月度评分模型，并将评分转换为组合和回测结果。

## 能力概览

- 研报 Markdown 解析、报告超边和股票关系边构建
- 行情、财务、行业和 Alpha 因子特征准备
- 可选的 LLM 关系抽取（`core_subject`、`peer`、`competition`、`positive_driver`、`risk_pressure`）
- 多模态研报 embedding 与股票图 GNN 评分
- TopK、TopK buffer、动量确认和日频执行策略
- 月度标签回测、日频买入/止损/换手控制和 HTML 报告
- 针对数据转换、图构建、预测和交易规则的 pytest 测试

## 目录结构

```text
alphagraph/
├── data/          行情、财务、研报和 Alpha 因子数据准备
├── kg/            研报解析与知识图谱关系抽取
├── graph/         股票图、月度样本和预测数据构建
├── cluster/       股票社区聚类
├── gnn/           GNN 训练和预测
├── model/         图模型、融合模块和训练逻辑
├── trading/       组合生成与交易策略
├── backtest/      回测执行与报告生成
├── tests/         pytest 测试
├── pipeline_config.toml
└── *.sh           各阶段命令行入口
```

## 环境准备

项目通过 Conda 环境运行。Shell 入口会读取 `pipeline_config.toml` 中的 `env_name`（当前默认值为 `rGNN`），可按本机环境修改。

```bash
conda activate rGNN
```

请在该环境中安装项目依赖；如果使用仓库约定的 `learn` 环境，可直接运行测试：

```bash
conda run -n learn python -m pytest tests -q
```

运行脚本前确认 `conda` 在 `PATH` 中，并确保当前工作目录是仓库的父目录，或使用脚本的绝对路径。

## 数据与凭据

默认数据路径在 `pipeline_config.toml` 中配置：

- 研报 Markdown：`reportdata/`
- Alpha 因子结果：`alpha/results/all_alpha_daily`
- 运行产物：`artifacts/alphagraph/<dataset_name>/`

大体量数据和生成产物不随代码仓库提交。运行前请准备对应的研报、行情和财务数据。

凭据必须通过环境变量或运行参数注入，不要写入配置文件或提交到 Git：

```bash
export TUSHARE_TOKEN='your-token'
export TUSHARE_TOKEN_URL='https://your-token-service.example/api'
export LLM_KG_API_KEY='your-api-key'
```

默认市场数据源是 `akshare`，因此普通流程不需要 Tushare 凭据；启用 Tushare 或 LLM KG 时才需要相应配置。

## 配置

所有 profile 位于 `pipeline_config.toml`。常用配置包括：

- `phase1_relation_aware`：关系感知图聚合主模型
- `rule_topk_buffer`：TopK 买入池与持有缓冲池
- `rule_momentum_confirmed`：高分候选股加动量过滤
- `rule_daily_execution`：月度选股加下月日频执行、止损和换手控制
- `strategy_all`：按配置组依次执行多个策略

可以先查看或修改 `[defaults]`、`[profiles.*]` 中的日期范围、数据目录、模型、训练轮数、选股数量和交易参数。

## 快速开始

从构建图谱到训练、回测的完整流程：

```bash
bash alphagraph/run_pipeline.sh phase1_relation_aware
```

日频执行策略：

```bash
bash alphagraph/run_pipeline.sh rule_daily_execution
```

运行全部策略配置：

```bash
bash alphagraph/run_pipeline.sh strategy_all
```

固定验证窗口（默认 `202503`~`202601`）：

```bash
bash alphagraph/run_fixed_window.sh
bash alphagraph/run_fixed_window.sh --dry-run strategy_all
```

完整流程依次执行：

1. 构建基础行情、财务、研报和图数据
2. 抽取 KG 关系并构建股票图
3. 训练 GNN 月度评分模型
4. 运行回测并生成指标、明细和 HTML 报告

## 分阶段运行

需要单独重跑某一阶段时：

```bash
bash alphagraph/build_graph.sh phase1_relation_aware
bash alphagraph/build_kg.sh artifacts/alphagraph/<dataset_name>
bash alphagraph/build_clusters.sh artifacts/alphagraph/<dataset_name>
bash alphagraph/train_gnn.sh phase1_relation_aware
bash alphagraph/backtest.sh phase1_relation_aware
```

`build_clusters.sh` 是否执行取决于 `cluster_enabled`。阶段脚本会从统一配置中读取环境、日期范围和模型参数。

## 生成预测

使用已配置的模型和数据生成指定月份的 TopK 推荐：

```bash
bash alphagraph/run_prediction.sh phase1_relation_aware \
  --dataset-name 202401_202607 \
  --signal-month 202607 \
  --as-of-date 2026-07-31 \
  --topk 20
```

预测脚本支持 `--force-retrain`、`--train-start-month`、`--train-end-month` 和 `--checkpoint-cadence-months` 等参数。模型 checkpoint 和推荐结果写入对应 artifact 目录。

## 直接运行回测

当评分和输入产物已经存在时，可直接调用回测模块：

```bash
conda run -n learn python -m alphagraph.backtest.runner \
  --scores-path artifacts/alphagraph/<dataset_name>/monthly_scores.parquet \
  --labels-path artifacts/alphagraph/<dataset_name>/monthly_labels.parquet \
  --tradability-path artifacts/alphagraph/<dataset_name>/monthly_stock_features.parquet \
  --benchmark-path artifacts/alphagraph/<dataset_name>/monthly_benchmark_returns.parquet \
  --daily-k-path artifacts/alphagraph/<dataset_name>/daily_k.parquet \
  --execution-mode daily \
  --trading-strategy topk_buffer \
  --entry-trigger pullback_or_breakout \
  --entry-window-days 15 \
  --buy-topk 20 \
  --hold-topk 50 \
  --max-positions 20
```

`monthly_label` 模式按月度标签直接结算；`daily` 模式使用日 K 线执行买入、涨跌停检查、止损、滑点和换手控制。

## 主要交易参数

- `trading_strategy`：`topk`、`topk_buffer` 或 `momentum_confirmed`
- `topk`、`buy_topk`、`hold_topk`、`max_positions`：组合规模与排名阈值
- `execution_mode`：`monthly_label` 或 `daily`
- `entry_trigger`：`immediate`、`pullback`、`breakout` 或 `pullback_or_breakout`
- `entry_window_days`：信号后等待买点的交易日数
- `stop_loss_pct`、`trailing_stop_pct`：止损和移动止盈
- `slippage_bps`、`max_monthly_turnover`：交易成本与换手约束
- `liquidity_quantile`、`min_price`、`exclude_st`：股票池过滤

## 产物

默认产物目录为：

```text
artifacts/alphagraph/<dataset_name>/
├── monthly_stock_features.parquet
├── monthly_labels.parquet
├── stock_graph_edges.parquet
├── report_llm_relations.parquet       # 启用 LLM KG 时生成
├── monthly_scores_<config>.parquet
└── backtest_runs/<run_tag>/
    ├── backtest_metrics.json
    ├── backtest_returns.parquet
    └── *.html
```

这些文件通常由流程生成，不建议手工编辑或提交到代码仓库。

## 测试

运行全部 AlphaGraph 测试：

```bash
conda run -n learn python -m pytest tests -q
```

只运行预测、图构建和交易相关测试：

```bash
conda run -n learn python -m pytest \
  tests/test_prediction_checkpoint.py \
  tests/test_prediction_data.py \
  tests/test_prediction_pipeline.py \
  tests/test_ablation_runner.py \
  tests/test_backtest_report_responsive.py -q
```

## 开发约定

- Python 模块使用 4 空格缩进，函数和变量使用 `snake_case`。
- 新逻辑优先放入职责明确的 `data/`、`kg/`、`graph/`、`gnn/`、`trading/` 或 `backtest/` 子模块。
- 测试放在 `tests/`，文件名使用 `test_*.py`。
- 不要提交 `artifacts/`、研报原始数据、模型缓存或任何 API token。
