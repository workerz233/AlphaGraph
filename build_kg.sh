#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "错误：未安装 conda，或 conda 不在 PATH 中。" >&2
  exit 1
fi
source "$SCRIPT_DIR/load_conda_env.sh"
load_conda_env "$SCRIPT_DIR/pipeline_config.toml"
eval "$(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" export --mode kg)"

# 产物目录：默认读取 pipeline_config.toml 中的 ARTIFACT_NAMESPACE 和 DATASET_NAME。
# 也可以通过第一个命令行参数显式指定，例如：bash alphagraph/build_kg.sh artifacts/alphagraph/202401_202601
ARTIFACT_DIR="${1:-artifacts/$ARTIFACT_NAMESPACE/$DATASET_NAME}"

# 是否强制重新调用 LLM 抽取关系：默认复用已有 report_llm_relations.parquet。
# 设置 FORCE_KG_EXTRACT=true 时，会忽略已有关系文件并重新抽取。
FORCE_KG_EXTRACT="${FORCE_KG_EXTRACT:-false}"
EXTRA_ARGS=()
case "$(printf '%s' "$FORCE_KG_EXTRACT" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    EXTRA_ARGS+=(--force-extract)
    ;;
esac

cd "$PROJECT_ROOT"

for required_path in \
  "$ARTIFACT_DIR/reports.parquet" \
  "$ARTIFACT_DIR/incidence.parquet" \
  "$ARTIFACT_DIR/monthly_stock_features.parquet" \
  "$ARTIFACT_DIR/daily_k.parquet" \
  "$ARTIFACT_DIR/hs300_daily_k.parquet"; do
  if [[ ! -f "$required_path" ]]; then
    echo "错误：未找到必需的 KG 输入： $required_path" >&2
    echo "请运行：bash alphagraph/build_graph.sh" >&2
    exit 1
  fi
done

if [[ "$MODEL_NAME" == /* ]] && [[ ! -d "$MODEL_NAME" ]]; then
  echo "警告：本地 MODEL_NAME 路径不存在： $MODEL_NAME" >&2
  echo "将回退到 ${MODEL_NAME_FALLBACK}，并启用 ModelScope 模型加载。" >&2
  MODEL_NAME="$MODEL_NAME_FALLBACK"
  export EMBEDDING_USE_MODELSCOPE=1
fi

echo "正在检查 conda 环境中的 KG 依赖： ${ENV_NAME}"
conda run --no-capture-output -n "$ENV_NAME" python -c "import pandas, vllm, modelscope"

echo "正在构建独立 alphagraph KG 产物..."
conda run --no-capture-output -n "$ENV_NAME" python -u -m alphagraph.kg.build \
  --reports-path "$ARTIFACT_DIR/reports.parquet" \
  --incidence-path "$ARTIFACT_DIR/incidence.parquet" \
  --monthly-features-path "$ARTIFACT_DIR/monthly_stock_features.parquet" \
  --daily-k-path "$ARTIFACT_DIR/daily_k.parquet" \
  --hs300-daily-k-path "$ARTIFACT_DIR/hs300_daily_k.parquet" \
  --report-embeddings-path "$ARTIFACT_DIR/report_embeddings.parquet" \
  --report-llm-relations-path "$ARTIFACT_DIR/report_llm_relations.parquet" \
  --kg-stock-edges-llm-path "$ARTIFACT_DIR/kg_stock_edges_llm.parquet" \
  --stock-graph-edges-path "$ARTIFACT_DIR/stock_graph_edges.parquet" \
  --kg-html-path "$ARTIFACT_DIR/kg_graph.html" \
  --model-name "$MODEL_NAME" \
  --stock-edge-builders "$STOCK_EDGE_BUILDERS" \
  --market-corr-lookback-days "$MARKET_CORR_LOOKBACK_DAYS" \
  --market-corr-topk "$MARKET_CORR_TOPK" \
  --market-corr-min-score "$MARKET_CORR_MIN_SCORE" \
  --industry-topk "$INDUSTRY_TOPK" \
  --llm-kg-enabled "$LLM_KG_ENABLED" \
  --llm-kg-base-url "$LLM_KG_BASE_URL" \
  --llm-kg-model "$LLM_KG_MODEL" \
  --llm-kg-max-chars "$LLM_KG_MAX_CHARS" \
  --llm-kg-timeout-seconds "$LLM_KG_TIMEOUT_SECONDS" \
  --llm-kg-min-confidence "$LLM_KG_MIN_CONFIDENCE" \
  --llm-kg-cache-dir "$LLM_KG_CACHE_DIR" \
  --llm-kg-raw-log-dir "$LLM_KG_RAW_LOG_DIR" \
  --llm-kg-decay-months "$LLM_KG_DECAY_MONTHS" \
  "${EXTRA_ARGS[@]}"

echo "完成，KG 输出写入："
echo "  $ARTIFACT_DIR/report_embeddings.parquet"
echo "  $ARTIFACT_DIR/report_llm_relations.parquet"
echo "  $ARTIFACT_DIR/kg_stock_edges_llm.parquet"
echo "  $ARTIFACT_DIR/stock_graph_edges.parquet"
echo "  $ARTIFACT_DIR/kg_graph.html"
