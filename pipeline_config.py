from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tomllib
from typing import Any


CONFIG_PATH = Path(__file__).with_name("pipeline_config.toml")
ENV_KEYS = {
    "alpha_factor_results_dir": "ALPHA_FACTOR_RESULTS_DIR",
    "alpha_factor_top_n": "ALPHA_FACTOR_TOP_N",
    "artifact_namespace": "ARTIFACT_NAMESPACE",
    "config_name": "CONFIG_NAME",
    "cluster_candidate_topk": "CLUSTER_CANDIDATE_TOPK",
    "cluster_edge_type_weights": "CLUSTER_EDGE_TYPE_WEIGHTS",
    "cluster_enabled": "CLUSTER_ENABLED",
    "cluster_max_per_cluster": "CLUSTER_MAX_PER_CLUSTER",
    "cluster_method": "CLUSTER_METHOD",
    "cluster_resolution": "CLUSTER_RESOLUTION",
    "community_reg_weight": "COMMUNITY_REG_WEIGHT",
    "cost_bps": "COST_BPS",
    "decay_months": "DECAY_MONTHS",
    "breakout_lookback_days": "BREAKOUT_LOOKBACK_DAYS",
    "breakout_volume_multiplier": "BREAKOUT_VOLUME_MULTIPLIER",
    "embedding_vllm_gpu_memory_utilization": "EMBEDDING_VLLM_GPU_MEMORY_UTILIZATION",
    "embedding_vllm_max_model_len": "EMBEDDING_VLLM_MAX_MODEL_LEN",
    "experiment_description": "EXPERIMENT_DESCRIPTION",
    "experiment_name": "EXPERIMENT_NAME",
    "env_name": "ENV_NAME",
    "epochs": "EPOCHS",
    "edge_type_prior_mode": "EDGE_TYPE_PRIOR_MODE",
    "exclude_st": "EXCLUDE_ST",
    "entry_delay_days": "ENTRY_DELAY_DAYS",
    "entry_trigger": "ENTRY_TRIGGER",
    "entry_window_days": "ENTRY_WINDOW_DAYS",
    "execution_mode": "EXECUTION_MODE",
    "fusion_type": "FUSION_TYPE",
    "graph_module_type": "GRAPH_MODULE_TYPE",
    "graph_snapshot_mode": "GRAPH_SNAPSHOT_MODE",
    "industry_topk": "INDUSTRY_TOPK",
    "llm_kg_enabled": "LLM_KG_ENABLED",
    "llm_kg_base_url": "LLM_KG_BASE_URL",
    "llm_kg_api_key": "LLM_KG_API_KEY",
    "llm_kg_model": "LLM_KG_MODEL",
    "llm_kg_max_chars": "LLM_KG_MAX_CHARS",
    "llm_kg_timeout_seconds": "LLM_KG_TIMEOUT_SECONDS",
    "llm_kg_min_confidence": "LLM_KG_MIN_CONFIDENCE",
    "llm_kg_cache_dir": "LLM_KG_CACHE_DIR",
    "llm_kg_raw_log_dir": "LLM_KG_RAW_LOG_DIR",
    "llm_kg_decay_months": "LLM_KG_DECAY_MONTHS",
    "buy_topk": "BUY_TOPK",
    "hold_topk": "HOLD_TOPK",
    "liquidity_quantile": "LIQUIDITY_QUANTILE",
    "markdown_dir": "MARKDOWN_DIR",
    "markdown_root": "MARKDOWN_ROOT",
    "market_corr_lookback_days": "MARKET_CORR_LOOKBACK_DAYS",
    "market_corr_min_score": "MARKET_CORR_MIN_SCORE",
    "market_corr_topk": "MARKET_CORR_TOPK",
    "market_data_source": "MARKET_DATA_SOURCE",
    "max_positions": "MAX_POSITIONS",
    "min_price": "MIN_PRICE",
    "model_name": "MODEL_NAME",
    "model_name_fallback": "MODEL_NAME_FALLBACK",
    "limit_threshold_pct": "LIMIT_THRESHOLD_PCT",
    "max_monthly_turnover": "MAX_MONTHLY_TURNOVER",
    "single_stock_max_weight": "SINGLE_STOCK_MAX_WEIGHT",
    "max_pullback_pct": "MAX_PULLBACK_PCT",
    "min_pullback_pct": "MIN_PULLBACK_PCT",
    "momentum_candidate_topk": "MOMENTUM_CANDIDATE_TOPK",
    "momentum_final_topk": "MOMENTUM_FINAL_TOPK",
    "momentum_liquidity_quantile": "MOMENTUM_LIQUIDITY_QUANTILE",
    "require_positive_ret_20d": "REQUIRE_POSITIVE_RET_20D",
    "require_positive_ret_60d": "REQUIRE_POSITIVE_RET_60D",
    "slippage_bps": "SLIPPAGE_BPS",
    "stop_loss_pct": "STOP_LOSS_PCT",
    "stock_edge_builders": "STOCK_EDGE_BUILDERS",
    "stock_universe": "STOCK_UNIVERSE",
    "topk": "TOPK",
    "trailing_stop_pct": "TRAILING_STOP_PCT",
    "trading_strategy": "TRADING_STRATEGY",
    "training_objective": "TRAINING_OBJECTIVE",
    "training_label": "TRAINING_LABEL",
    "tushare_token_url": "TUSHARE_TOKEN_URL",
    "scores_path": "SCORES_PATH",
    "start_month": "START_MONTH",
    "end_month": "END_MONTH",
    "dataset_name": "DATASET_NAME",
    "single_month_mode": "SINGLE_MONTH_MODE",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read alphagraph pipeline configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument(
        "--mode",
        choices=("pipeline", "graph", "gnn", "kg", "backtest", "train", "test"),
        default="pipeline",
    )
    export_parser.add_argument("--config-name", default=None)

    is_name_parser = subparsers.add_parser("is-name")
    is_name_parser.add_argument("name")

    group_parser = subparsers.add_parser("group-members")
    group_parser.add_argument("name")

    args = parser.parse_args()
    config = _read_config()

    if args.command == "export":
        values = resolve_config(config, mode=args.mode, config_name=args.config_name)
        for key in sorted(values):
            print(f'{key}="{_escape_shell_double_quoted(str(values[key]))}"')
        return

    if args.command == "is-name":
        if is_config_name(config, args.name):
            return
        sys.exit(1)

    if args.command == "group-members":
        members = profile_group_members(config, args.name)
        for member in members:
            print(member)
        return


def resolve_config(
    config: dict[str, Any],
    mode: str = "pipeline",
    config_name: str | None = None,
) -> dict[str, str]:
    mode_aliases = {"train": "gnn", "test": "backtest"}
    mode = mode_aliases.get(mode, mode)
    raw_values: dict[str, Any] = {}
    raw_values.update(config.get("defaults", {}))
    raw_values.update(config.get("mode_defaults", {}).get(mode, {}))

    selected_config_name = config_name or os.environ.get("CONFIG_NAME") or raw_values.get("config_name")
    if selected_config_name:
        raw_values["config_name"] = selected_config_name
        raw_values.update(config.get("profiles", {}).get(str(selected_config_name), {}))

    values: dict[str, str] = {}
    for raw_key, env_key in ENV_KEYS.items():
        if env_key == "CONFIG_NAME" and config_name is not None:
            values[env_key] = str(config_name)
        elif env_key in os.environ:
            values[env_key] = os.environ[env_key]
        elif raw_key in raw_values:
            values[env_key] = str(raw_values[raw_key])

    # 多月任务扫描研报根目录；单月任务仍使用具体目录，显式环境变量优先。
    if "MARKDOWN_DIR" not in os.environ:
        start_month = values.get("START_MONTH", "").strip()
        end_month = values.get("END_MONTH", "").strip()
        markdown_root = values.get("MARKDOWN_ROOT", "").strip()
        if start_month and end_month and start_month != end_month and markdown_root:
            values["MARKDOWN_DIR"] = markdown_root
    return values


def is_config_name(config: dict[str, Any], name: str) -> bool:
    return name in config.get("profiles", {}) or name in config.get("profile_groups", {})


def profile_group_members(config: dict[str, Any], name: str) -> list[str]:
    if name in config.get("profile_groups", {}):
        return list(config["profile_groups"][name])
    if name in config.get("profiles", {}):
        return [name]
    raise SystemExit(f"Unknown config name: {name}")


def _read_config() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _escape_shell_double_quoted(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )


if __name__ == "__main__":
    main()
