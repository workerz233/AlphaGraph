#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/load_conda_env.sh"
load_conda_env "$SCRIPT_DIR/pipeline_config.toml"

if ! command -v conda >/dev/null 2>&1; then
  echo "错误：未安装 conda，或 conda 不在 PATH 中。" >&2
  exit 1
fi

eval "$(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" export --mode backtest)"

if [[ $# -ge 1 ]]; then
  if conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" is-name "$1"; then
    CONFIG_NAME="$1"
    eval "$(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" export --mode backtest --config-name "$CONFIG_NAME")"
  else
    MARKDOWN_DIR="$1"
  fi
fi

if [[ $# -ge 2 ]]; then
  MARKDOWN_DIR="$2"
fi

ARTIFACT_DIR="$PROJECT_ROOT/artifacts/$ARTIFACT_NAMESPACE/$DATASET_NAME"
BACKTEST_RUN_TAG="${BACKTEST_RUN_TAG:-$(date '+%Y_%m_%d-%H%M%S')}"
SCORES_PATH="${SCORES_PATH:-$ARTIFACT_DIR/monthly_scores.parquet}"
BACKTEST_DIR="$ARTIFACT_DIR/backtest_runs/$BACKTEST_RUN_TAG"
DATASET_MONTH="${DATASET_NAME%%_*}"
REPORT_DATE="${BACKTEST_RUN_TAG:0:10}"
REPORT_DATE="${REPORT_DATE//_/}"
REPORT_HTML="$BACKTEST_DIR/${DATASET_MONTH}+${REPORT_DATE}.html"
EXTRA_CONFIG_ARGS=()
if [[ -n "$DECAY_MONTHS" ]]; then
  EXTRA_CONFIG_ARGS+=(--decay-months "$DECAY_MONTHS")
fi
if [[ -n "$EPOCHS" ]]; then
  EXTRA_CONFIG_ARGS+=(--epochs "$EPOCHS")
fi

cd "$PROJECT_ROOT"

for required_path in \
  "$SCORES_PATH" \
  "$ARTIFACT_DIR/monthly_labels.parquet" \
  "$ARTIFACT_DIR/monthly_benchmark_returns.parquet" \
  "$ARTIFACT_DIR/monthly_stock_features.parquet"; do
  if [[ ! -f "$required_path" ]]; then
    echo "错误：未找到回测所需输入： $required_path" >&2
    echo "请先运行缺失的上游模块，或使用：bash alphagraph/run_pipeline.sh <配置名>。" >&2
    exit 1
  fi
done

if [[ -e "$BACKTEST_DIR" ]]; then
  echo "错误：回测输出目录已存在： $BACKTEST_DIR" >&2
  echo "请设置新的 BACKTEST_RUN_TAG，避免覆盖已有结果。" >&2
  exit 1
fi
mkdir -p "$BACKTEST_DIR"

echo "正在检查 conda 环境中的回测依赖： ${ENV_NAME}"
conda run --no-capture-output -n "$ENV_NAME" python -c "import pandas"

echo "正在运行 alphagraph 回测..."
conda run --no-capture-output -n "$ENV_NAME" python -u -m alphagraph.backtest.runner \
  --scores-path "$SCORES_PATH" \
  --labels-path "$ARTIFACT_DIR/monthly_labels.parquet" \
  --tradability-path "$ARTIFACT_DIR/monthly_stock_features.parquet" \
  --benchmark-path "$ARTIFACT_DIR/monthly_benchmark_returns.parquet" \
  --output-returns "$BACKTEST_DIR/backtest_returns.parquet" \
  --output-metrics "$BACKTEST_DIR/backtest_metrics.json" \
  --output-html "$REPORT_HTML" \
  --stock-graph-edges-path "$ARTIFACT_DIR/stock_graph_edges.parquet" \
  --daily-k-path "$ARTIFACT_DIR/daily_k.parquet" \
  --k "$TOPK" \
  --cost-bps "$COST_BPS" \
  --trading-strategy "$TRADING_STRATEGY" \
  --buy-topk "$BUY_TOPK" \
  --hold-topk "$HOLD_TOPK" \
  --max-positions "$MAX_POSITIONS" \
  --liquidity-quantile "$LIQUIDITY_QUANTILE" \
  --min-price "$MIN_PRICE" \
  --exclude-st "$EXCLUDE_ST" \
  --momentum-candidate-topk "$MOMENTUM_CANDIDATE_TOPK" \
  --momentum-final-topk "$MOMENTUM_FINAL_TOPK" \
  --require-positive-ret-20d "$REQUIRE_POSITIVE_RET_20D" \
  --require-positive-ret-60d "$REQUIRE_POSITIVE_RET_60D" \
  --momentum-liquidity-quantile "$MOMENTUM_LIQUIDITY_QUANTILE" \
  --stock-clusters-path "$ARTIFACT_DIR/stock_clusters.parquet" \
  --cluster-candidate-topk "$CLUSTER_CANDIDATE_TOPK" \
  --cluster-max-per-cluster "$CLUSTER_MAX_PER_CLUSTER" \
  --execution-mode "$EXECUTION_MODE" \
  --entry-delay-days "$ENTRY_DELAY_DAYS" \
  --stop-loss-pct "$STOP_LOSS_PCT" \
  --trailing-stop-pct "$TRAILING_STOP_PCT" \
  --limit-threshold-pct "$LIMIT_THRESHOLD_PCT" \
  --slippage-bps "$SLIPPAGE_BPS" \
  --max-monthly-turnover "$MAX_MONTHLY_TURNOVER" \
  --single-stock-max-weight "$SINGLE_STOCK_MAX_WEIGHT" \
  --entry-trigger "$ENTRY_TRIGGER" \
  --entry-window-days "$ENTRY_WINDOW_DAYS" \
  --min-pullback-pct "$MIN_PULLBACK_PCT" \
  --max-pullback-pct "$MAX_PULLBACK_PCT" \
  --breakout-lookback-days "$BREAKOUT_LOOKBACK_DAYS" \
  --breakout-volume-multiplier "$BREAKOUT_VOLUME_MULTIPLIER" \
  --config-name "$CONFIG_NAME" \
  --experiment-name "${EXPERIMENT_NAME:-}" \
  --experiment-description "${EXPERIMENT_DESCRIPTION:-}" \
  --fusion-type "$FUSION_TYPE" \
  --model-name "$MODEL_NAME" \
  --training-label "$TRAINING_LABEL" \
  "${EXTRA_CONFIG_ARGS[@]}"

echo "完成，回测输出写入："
echo "  $BACKTEST_DIR/backtest_returns.parquet"
echo "  $BACKTEST_DIR/backtest_metrics.json"
echo "  $REPORT_HTML"
echo "  $PROJECT_ROOT/artifacts/alphagraph/index.html"
