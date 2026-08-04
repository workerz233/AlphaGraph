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

eval "$(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" export --mode kg)"

if [[ $# -ge 1 ]]; then
  if conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" is-name "$1"; then
    CONFIG_NAME="$1"
    eval "$(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" export --mode kg --config-name "$CONFIG_NAME")"
  else
    ARTIFACT_DIR="$1"
  fi
fi

ARTIFACT_DIR="${ARTIFACT_DIR:-$PROJECT_ROOT/artifacts/$ARTIFACT_NAMESPACE/$DATASET_NAME}"

cd "$PROJECT_ROOT"

for required_path in \
  "$ARTIFACT_DIR/monthly_stock_features.parquet" \
  "$ARTIFACT_DIR/stock_graph_edges.parquet"; do
  if [[ ! -f "$required_path" ]]; then
    echo "错误：未找到聚类所需输入： $required_path" >&2
    exit 1
  fi
done

echo "正在构建 alphagraph 股票社区聚类..."
conda run --no-capture-output -n "$ENV_NAME" python -u -m alphagraph.cluster.build \
  --monthly-features-path "$ARTIFACT_DIR/monthly_stock_features.parquet" \
  --stock-graph-edges-path "$ARTIFACT_DIR/stock_graph_edges.parquet" \
  --incidence-path "$ARTIFACT_DIR/incidence.parquet" \
  --output-clusters "$ARTIFACT_DIR/stock_clusters.parquet" \
  --output-cluster-edges "$ARTIFACT_DIR/cluster_graph_edges.parquet" \
  --cluster-method "$CLUSTER_METHOD" \
  --cluster-resolution "$CLUSTER_RESOLUTION" \
  --cluster-edge-type-weights "$CLUSTER_EDGE_TYPE_WEIGHTS"

echo "完成，聚类产物写入："
echo "  $ARTIFACT_DIR/stock_clusters.parquet"
echo "  $ARTIFACT_DIR/cluster_graph_edges.parquet"
