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

eval "$(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" export --mode graph)"

if [[ $# -ge 1 ]]; then
  if conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" is-name "$1"; then
    CONFIG_NAME="$1"
    eval "$(conda run -n "$ENV_NAME" python "$SCRIPT_DIR/pipeline_config.py" export --mode graph --config-name "$CONFIG_NAME")"
  else
    MARKDOWN_DIR="$1"
  fi
fi

if [[ $# -ge 2 ]]; then
  MARKDOWN_DIR="$2"
fi

if [[ ! -d "$PROJECT_ROOT/$MARKDOWN_DIR" ]]; then
  echo "错误：未找到研报目录：$PROJECT_ROOT/$MARKDOWN_DIR" >&2
  exit 1
fi

ARTIFACT_DIR="$PROJECT_ROOT/artifacts/$ARTIFACT_NAMESPACE/$DATASET_NAME"

cd "$PROJECT_ROOT"
mkdir -p "$ARTIFACT_DIR"

echo "正在检查 conda 环境中的基础建图依赖： ${ENV_NAME}"
if [[ "${MARKET_DATA_SOURCE:-akshare}" == "akshare" ]]; then
  conda run --no-capture-output -n "$ENV_NAME" python -c "import akshare, pandas"
elif [[ "${MARKET_DATA_SOURCE:-akshare}" == "tushare" ]]; then
  conda run --no-capture-output -n "$ENV_NAME" python -c "import tushare, pandas"
else
  conda run --no-capture-output -n "$ENV_NAME" python -c "import baostock, pandas"
fi

echo "正在为 ${MARKDOWN_DIR} 构建 alphagraph 基础数据产物..."
conda run --no-capture-output -n "$ENV_NAME" python -u -m alphagraph.graph.build \
  --workspace-root "$PROJECT_ROOT" \
  --markdown-dir "$MARKDOWN_DIR" \
  --start-month "$START_MONTH" \
  --end-month "$END_MONTH" \
  --artifact-dir "$ARTIFACT_DIR" \
  --stock-universe "$STOCK_UNIVERSE" \
  --market-data-source "$MARKET_DATA_SOURCE" \
  --tushare-token-url "$TUSHARE_TOKEN_URL" \
  --alpha-factor-results-dir "$ALPHA_FACTOR_RESULTS_DIR" \
  --alpha-factor-top-n "$ALPHA_FACTOR_TOP_N"

echo "完成，基础数据产物写入："
echo "  $ARTIFACT_DIR"
