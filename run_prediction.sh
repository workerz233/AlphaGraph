#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "用法：bash alphagraph/run_prediction.sh <配置名> [预测参数]" >&2
  exit 1
fi

CONFIG_NAME="$1"
shift

if ! command -v conda >/dev/null 2>&1; then
  echo "错误：未安装 conda，或 conda 不在 PATH 中。" >&2
  exit 1
fi
source "$SCRIPT_DIR/load_conda_env.sh"
load_conda_env "$SCRIPT_DIR/pipeline_config.toml"
eval "$(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" export --mode gnn --config-name "$CONFIG_NAME")"

MARKDOWN_DIR="${MARKDOWN_DIR:-reportdata}"
DATASET_NAME="${DATASET_NAME:-202401_202607}"
DATA_START_MONTH="${DATA_START_MONTH:-202401}"
SIGNAL_MONTH="${SIGNAL_MONTH:-202607}"
AS_OF_DATE="${AS_OF_DATE:-2026-07-31}"
TRAIN_START_MONTH="${TRAIN_START_MONTH:-202507}"
TRAIN_END_MONTH="${TRAIN_END_MONTH:-202606}"
CHECKPOINT_CADENCE_MONTHS="${CHECKPOINT_CADENCE_MONTHS:-3}"
TOPK="${TOPK:-20}"
FORCE_RETRAIN="${FORCE_RETRAIN:-false}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --markdown-dir) MARKDOWN_DIR="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --data-start-month) DATA_START_MONTH="$2"; shift 2 ;;
    --signal-month) SIGNAL_MONTH="$2"; shift 2 ;;
    --as-of-date) AS_OF_DATE="$2"; shift 2 ;;
    --train-start-month) TRAIN_START_MONTH="$2"; shift 2 ;;
    --train-end-month) TRAIN_END_MONTH="$2"; shift 2 ;;
    --checkpoint-cadence-months) CHECKPOINT_CADENCE_MONTHS="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --force-retrain) FORCE_RETRAIN=true; shift ;;
    *) echo "错误：未知参数 $1" >&2; exit 2 ;;
  esac
done

if [[ "$LLM_KG_ENABLED" =~ ^([Tt]rue|1|yes|YES)$ ]] && [[ -z "$LLM_KG_API_KEY" ]]; then
  echo "错误：LLM KG 已启用，请先在服务器设置 LLM_KG_API_KEY。" >&2
  exit 1
fi

ARTIFACT_DIR="$PROJECT_ROOT/artifacts/$ARTIFACT_NAMESPACE/$DATASET_NAME"
mkdir -p "$ARTIFACT_DIR"
cd "$PROJECT_ROOT"

if [[ "$MODEL_NAME" == /* ]] && [[ ! -d "$MODEL_NAME" ]]; then
  echo "警告：本地 MODEL_NAME 路径不存在： $MODEL_NAME" >&2
  echo "将使用 fallback 模型：$MODEL_NAME_FALLBACK" >&2
  MODEL_NAME="$MODEL_NAME_FALLBACK"
  export EMBEDDING_USE_MODELSCOPE=1
fi

echo "[1/3] 构建截至 $AS_OF_DATE 的预测数据..."
conda run --no-capture-output -n "$ENV_NAME" python -u -m alphagraph.graph.build \
  --workspace-root "$PROJECT_ROOT" \
  --markdown-dir "$MARKDOWN_DIR" \
  --artifact-dir "$ARTIFACT_DIR" \
  --start-month "$DATA_START_MONTH" \
  --end-month "$SIGNAL_MONTH" \
  --data-start-month "$DATA_START_MONTH" \
  --signal-month "$SIGNAL_MONTH" \
  --as-of-date "$AS_OF_DATE" \
  --prediction \
  --stock-universe "$STOCK_UNIVERSE" \
  --market-data-source "$MARKET_DATA_SOURCE" \
  --tushare-token-url "$TUSHARE_TOKEN_URL"

echo "[2/3] 构建研报 embedding 和股票关系图..."
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
bash "$SCRIPT_DIR/build_kg.sh" "$ARTIFACT_DIR"

echo "[3/3] 训练或加载季度 checkpoint，并生成推荐..."
predict_args=(
  --artifact-dir "$ARTIFACT_DIR"
  --signal-month "$SIGNAL_MONTH"
  --as-of-date "$AS_OF_DATE"
  --train-start-month "$TRAIN_START_MONTH"
  --train-end-month "$TRAIN_END_MONTH"
  --checkpoint-cadence-months "$CHECKPOINT_CADENCE_MONTHS"
  --topk "$TOPK"
  --epochs "$EPOCHS"
  --decay-months "$DECAY_MONTHS"
  --fusion-type "$FUSION_TYPE"
  --graph-module-type "$GRAPH_MODULE_TYPE"
  --edge-type-prior-mode "$EDGE_TYPE_PRIOR_MODE"
  --community-reg-weight "$COMMUNITY_REG_WEIGHT"
  --training-objective "$TRAINING_OBJECTIVE"
  --training-label "$TRAINING_LABEL"
  --liquidity-quantile "$LIQUIDITY_QUANTILE"
  --min-price "$MIN_PRICE"
  --exclude-st "$EXCLUDE_ST"
  --model-name "$MODEL_NAME"
)
if [[ "$FORCE_RETRAIN" =~ ^([Tt]rue|1|yes|YES)$ ]]; then
  predict_args+=(--force-retrain)
fi
conda run --no-capture-output -n "$ENV_NAME" python -u -m alphagraph.gnn.predict "${predict_args[@]}"
