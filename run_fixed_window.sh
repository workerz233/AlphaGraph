#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
用法：
  bash alphagraph/run_fixed_window.sh [--dry-run] [配置名或配置组]

默认配置名：
  phase1_relation_aware

示例：
  bash alphagraph/run_fixed_window.sh
  bash alphagraph/run_fixed_window.sh rule_daily_execution
  bash alphagraph/run_fixed_window.sh strategy_all
  bash alphagraph/run_fixed_window.sh --dry-run strategy_all

该脚本固定默认验证区间为 202503~202601，并写入：
  artifacts/alphagraph/202503_202601/

可通过环境变量覆盖：
  MARKDOWN_DIR, START_MONTH, END_MONTH, DATASET_NAME, MARKET_DATA_SOURCE
EOF
}

dry_run=false
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

config_name="${1:-phase1_relation_aware}"

export MARKDOWN_DIR="${MARKDOWN_DIR:-reportdata}"
export START_MONTH="${START_MONTH:-202503}"
export END_MONTH="${END_MONTH:-202601}"
export DATASET_NAME="${DATASET_NAME:-202503_202601}"
export MARKET_DATA_SOURCE="${MARKET_DATA_SOURCE:-akshare}"

echo "准备运行 alphagraph："
echo "  CONFIG_NAME=$config_name"
echo "  MARKDOWN_DIR=$MARKDOWN_DIR"
echo "  MARKET_DATA_SOURCE=$MARKET_DATA_SOURCE"
echo "  START_MONTH=$START_MONTH"
echo "  END_MONTH=$END_MONTH"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  ARTIFACT_DIR=$PROJECT_ROOT/artifacts/alphagraph/$DATASET_NAME"

if [[ "$dry_run" == "true" ]]; then
  echo "dry-run：未启动 pipeline。"
  echo "将执行：bash $SCRIPT_DIR/run_pipeline.sh $config_name"
  exit 0
fi

exec bash "$SCRIPT_DIR/run_pipeline.sh" "$config_name"
