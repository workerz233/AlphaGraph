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

eval "$(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" export --mode pipeline)"

if [[ $# -lt 1 ]]; then
  echo "用法：bash alphagraph/run_pipeline.sh <配置名>" >&2
  exit 1
fi

config_name_or_dir="$1"
if ! conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" is-name "$config_name_or_dir"; then
  echo "错误：run_pipeline.sh 需要配置名，当前输入为：$config_name_or_dir" >&2
  echo "如需单独运行模块，请使用 build_graph.sh、build_kg.sh、train_gnn.sh 或 backtest.sh。" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

next_backtest_run_tag() {
  local artifact_dir="$1"
  local candidate
  while true; do
    candidate="$(date '+%Y_%m_%d-%H%M%S')"
    if [[ ! -e "$artifact_dir/backtest_runs/$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
    sleep 1
  done
}

run_one_config() {
  local config_name="$1"
  eval "$(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" export --mode pipeline --config-name "$config_name")"

  local artifact_dir="$PROJECT_ROOT/artifacts/$ARTIFACT_NAMESPACE/$DATASET_NAME"
  local scores_path="$artifact_dir/monthly_scores_${config_name}.parquet"
  local backtest_run_tag
  backtest_run_tag="$(next_backtest_run_tag "$artifact_dir")"
  local backtest_dir="$artifact_dir/backtest_runs/$backtest_run_tag"

  echo "[1/4][$config_name] 正在构建 alphagraph 基础数据产物..."
  ENV_NAME="$ENV_NAME" \
  STOCK_UNIVERSE="$STOCK_UNIVERSE" \
  MARKET_DATA_SOURCE="$MARKET_DATA_SOURCE" \
  TUSHARE_TOKEN_URL="$TUSHARE_TOKEN_URL" \
  bash "$SCRIPT_DIR/build_graph.sh" "$config_name_or_dir"

  echo "[2/4][$config_name] 正在构建 alphagraph KG 和完整股票图..."
  ENV_NAME="$ENV_NAME" \
  MODEL_NAME="$MODEL_NAME" \
  MODEL_NAME_FALLBACK="$MODEL_NAME_FALLBACK" \
  STOCK_EDGE_BUILDERS="$STOCK_EDGE_BUILDERS" \
  MARKET_CORR_LOOKBACK_DAYS="$MARKET_CORR_LOOKBACK_DAYS" \
  MARKET_CORR_TOPK="$MARKET_CORR_TOPK" \
  MARKET_CORR_MIN_SCORE="$MARKET_CORR_MIN_SCORE" \
  INDUSTRY_TOPK="$INDUSTRY_TOPK" \
  LLM_KG_ENABLED="$LLM_KG_ENABLED" \
  LLM_KG_BASE_URL="$LLM_KG_BASE_URL" \
  LLM_KG_API_KEY="$LLM_KG_API_KEY" \
  LLM_KG_MODEL="$LLM_KG_MODEL" \
  LLM_KG_MAX_CHARS="$LLM_KG_MAX_CHARS" \
  LLM_KG_TIMEOUT_SECONDS="$LLM_KG_TIMEOUT_SECONDS" \
  LLM_KG_MIN_CONFIDENCE="$LLM_KG_MIN_CONFIDENCE" \
  LLM_KG_CACHE_DIR="$LLM_KG_CACHE_DIR" \
  LLM_KG_RAW_LOG_DIR="$LLM_KG_RAW_LOG_DIR" \
  LLM_KG_DECAY_MONTHS="$LLM_KG_DECAY_MONTHS" \
  EMBEDDING_VLLM_GPU_MEMORY_UTILIZATION="$EMBEDDING_VLLM_GPU_MEMORY_UTILIZATION" \
  EMBEDDING_VLLM_MAX_MODEL_LEN="$EMBEDDING_VLLM_MAX_MODEL_LEN" \
  bash "$SCRIPT_DIR/build_kg.sh" "$artifact_dir"

  if [[ "${CLUSTER_ENABLED:-true}" =~ ^([Tt]rue|1|yes|YES)$ ]]; then
    echo "[2b/4][$config_name] 正在构建 alphagraph 股票社区聚类..."
    ENV_NAME="$ENV_NAME" \
    ARTIFACT_NAMESPACE="$ARTIFACT_NAMESPACE" \
    DATASET_NAME="$DATASET_NAME" \
    CLUSTER_METHOD="$CLUSTER_METHOD" \
    CLUSTER_RESOLUTION="$CLUSTER_RESOLUTION" \
    CLUSTER_EDGE_TYPE_WEIGHTS="$CLUSTER_EDGE_TYPE_WEIGHTS" \
    bash "$SCRIPT_DIR/build_clusters.sh" "$artifact_dir"
  fi

  echo "[3/4][$config_name] 正在训练 alphagraph GNN..."
  ENV_NAME="$ENV_NAME" \
  MODEL_NAME="$MODEL_NAME" \
  START_MONTH="$START_MONTH" \
  END_MONTH="$END_MONTH" \
  DECAY_MONTHS="$DECAY_MONTHS" \
  EPOCHS="$EPOCHS" \
  FUSION_TYPE="$FUSION_TYPE" \
  GRAPH_MODULE_TYPE="$GRAPH_MODULE_TYPE" \
  EDGE_TYPE_PRIOR_MODE="$EDGE_TYPE_PRIOR_MODE" \
  COMMUNITY_REG_WEIGHT="$COMMUNITY_REG_WEIGHT" \
  TRAINING_LABEL="$TRAINING_LABEL" \
  SCORES_PATH="$scores_path" \
  bash "$SCRIPT_DIR/train_gnn.sh" "$config_name_or_dir"

  echo "[4/4][$config_name] 正在运行 alphagraph 回测..."
  ENV_NAME="$ENV_NAME" \
  TOPK="$TOPK" \
  COST_BPS="$COST_BPS" \
  START_MONTH="$START_MONTH" \
  END_MONTH="$END_MONTH" \
  SCORES_PATH="$scores_path" \
  BACKTEST_RUN_TAG="$backtest_run_tag" \
  CONFIG_NAME="$config_name" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-}" \
  EXPERIMENT_DESCRIPTION="${EXPERIMENT_DESCRIPTION:-}" \
  FUSION_TYPE="$FUSION_TYPE" \
  GRAPH_MODULE_TYPE="$GRAPH_MODULE_TYPE" \
  TRAINING_LABEL="$TRAINING_LABEL" \
  MODEL_NAME="$MODEL_NAME" \
  DECAY_MONTHS="$DECAY_MONTHS" \
  EPOCHS="$EPOCHS" \
  TRADING_STRATEGY="$TRADING_STRATEGY" \
  BUY_TOPK="$BUY_TOPK" \
  HOLD_TOPK="$HOLD_TOPK" \
  MAX_POSITIONS="$MAX_POSITIONS" \
  LIQUIDITY_QUANTILE="$LIQUIDITY_QUANTILE" \
  MIN_PRICE="$MIN_PRICE" \
  EXCLUDE_ST="$EXCLUDE_ST" \
  MOMENTUM_CANDIDATE_TOPK="$MOMENTUM_CANDIDATE_TOPK" \
  MOMENTUM_FINAL_TOPK="$MOMENTUM_FINAL_TOPK" \
  REQUIRE_POSITIVE_RET_20D="$REQUIRE_POSITIVE_RET_20D" \
  REQUIRE_POSITIVE_RET_60D="$REQUIRE_POSITIVE_RET_60D" \
  MOMENTUM_LIQUIDITY_QUANTILE="$MOMENTUM_LIQUIDITY_QUANTILE" \
  CLUSTER_CANDIDATE_TOPK="$CLUSTER_CANDIDATE_TOPK" \
  CLUSTER_MAX_PER_CLUSTER="$CLUSTER_MAX_PER_CLUSTER" \
  EXECUTION_MODE="$EXECUTION_MODE" \
  ENTRY_DELAY_DAYS="$ENTRY_DELAY_DAYS" \
  STOP_LOSS_PCT="$STOP_LOSS_PCT" \
  TRAILING_STOP_PCT="$TRAILING_STOP_PCT" \
  LIMIT_THRESHOLD_PCT="$LIMIT_THRESHOLD_PCT" \
  SLIPPAGE_BPS="$SLIPPAGE_BPS" \
  MAX_MONTHLY_TURNOVER="$MAX_MONTHLY_TURNOVER" \
  ENTRY_TRIGGER="$ENTRY_TRIGGER" \
  ENTRY_WINDOW_DAYS="$ENTRY_WINDOW_DAYS" \
  MIN_PULLBACK_PCT="$MIN_PULLBACK_PCT" \
  MAX_PULLBACK_PCT="$MAX_PULLBACK_PCT" \
  BREAKOUT_LOOKBACK_DAYS="$BREAKOUT_LOOKBACK_DAYS" \
  BREAKOUT_VOLUME_MULTIPLIER="$BREAKOUT_VOLUME_MULTIPLIER" \
  bash "$SCRIPT_DIR/backtest.sh" "$config_name_or_dir"

  echo "配置 ${config_name} 已完成："
  echo "  $scores_path"
  echo "  $artifact_dir/kg_graph.html"
  echo "  $backtest_dir/backtest_returns.parquet"
  echo "  $backtest_dir/backtest_metrics.json"
  local dataset_month="${DATASET_NAME%%_*}"
  local report_date="${backtest_run_tag:0:10}"
  report_date="${report_date//_/}"
  echo "  $backtest_dir/${dataset_month}+${report_date}.html"
  echo "  $PROJECT_ROOT/artifacts/alphagraph/index.html"
}

config_count="$(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" group-members "$config_name_or_dir" | wc -l | tr -d ' ')"
if [[ "$config_count" -gt 1 ]]; then
  while IFS= read -r config_name; do
    config_name_or_dir="$config_name"
    run_one_config "$config_name"
  done < <(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" group-members "$1")
else
  run_one_config "$config_name_or_dir"
fi
