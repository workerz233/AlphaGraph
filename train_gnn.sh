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

eval "$(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" export --mode gnn)"

if [[ $# -ge 1 ]]; then
  if conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" is-name "$1"; then
    CONFIG_NAME="$1"
    eval "$(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" export --mode gnn --config-name "$CONFIG_NAME")"
  else
    MARKDOWN_DIR="$1"
  fi
fi

if [[ $# -ge 2 ]]; then
  MARKDOWN_DIR="$2"
fi

ARTIFACT_DIR="$PROJECT_ROOT/artifacts/$ARTIFACT_NAMESPACE/$DATASET_NAME"
SCORES_PATH="${SCORES_PATH:-$ARTIFACT_DIR/monthly_scores.parquet}"

cd "$PROJECT_ROOT"

for required_path in \
  "$ARTIFACT_DIR/reports.parquet" \
  "$ARTIFACT_DIR/incidence.parquet" \
  "$ARTIFACT_DIR/monthly_stock_features.parquet" \
  "$ARTIFACT_DIR/monthly_labels.parquet" \
  "$ARTIFACT_DIR/report_embeddings.parquet" \
  "$ARTIFACT_DIR/stock_graph_edges.parquet"; do
  if [[ ! -f "$required_path" ]]; then
    echo "错误：未找到训练所需图产物： $required_path" >&2
    echo "请先依次运行：bash alphagraph/build_graph.sh，然后运行 bash alphagraph/build_kg.sh" >&2
    exit 1
  fi
done

echo "正在检查 conda 环境中的 GNN 依赖： ${ENV_NAME}"
conda run --no-capture-output -n "$ENV_NAME" python -c "import pandas, torch"

echo "正在训练 alphagraph GNN 选股模型..."
conda run --no-capture-output -n "$ENV_NAME" python -u -m alphagraph.gnn.train \
  --reports-path "$ARTIFACT_DIR/reports.parquet" \
  --incidence-path "$ARTIFACT_DIR/incidence.parquet" \
  --monthly-features-path "$ARTIFACT_DIR/monthly_stock_features.parquet" \
  --labels-path "$ARTIFACT_DIR/monthly_labels.parquet" \
  --embeddings-path "$ARTIFACT_DIR/report_embeddings.parquet" \
  --stock-graph-edges-path "$ARTIFACT_DIR/stock_graph_edges.parquet" \
  --scores-path "$SCORES_PATH" \
  --model-name "$MODEL_NAME" \
  --decay-months "$DECAY_MONTHS" \
  --epochs "$EPOCHS" \
  --fusion-type "$FUSION_TYPE" \
  --graph-module-type "$GRAPH_MODULE_TYPE" \
  --edge-type-prior-mode "$EDGE_TYPE_PRIOR_MODE" \
  --training-objective "$TRAINING_OBJECTIVE" \
  --training-label "$TRAINING_LABEL" \
  --stock-clusters-path "$ARTIFACT_DIR/stock_clusters.parquet" \
  --cluster-graph-edges-path "$ARTIFACT_DIR/cluster_graph_edges.parquet" \
  --community-reg-weight "$COMMUNITY_REG_WEIGHT"

echo "完成，GNN 输出写入："
echo "  $SCORES_PATH"
