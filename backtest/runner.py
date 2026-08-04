from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from alphagraph.data.market_data import FORWARD_RETURN_HORIZONS
from alphagraph.kg.report import build_kg_graph_payload as _shared_build_kg_graph_payload
from alphagraph.trading.strategies import build_portfolio_by_strategy, run_daily_signal_backtest

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2] / "artifacts" / "alphagraph"
)


def _default_backtest_report_path(run_dir: Path) -> Path:
    dataset = run_dir.parents[1].name
    dataset_month = _extract_dataset_month(dataset)
    run_date = run_dir.name.split("-", 1)[0].replace("_", "")
    return run_dir / f"{dataset_month}+{run_date}.html"


def run_monthly_backtest(
    scores_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    cost_bps: float = 20,
    k: int = 20,
    tradability_df: pd.DataFrame | None = None,
    trading_strategy: str = "topk",
    buy_topk: int = 20,
    hold_topk: int = 50,
    max_positions: int = 20,
    liquidity_quantile: float = 0.2,
    min_price: float = 2.0,
    exclude_st: bool = True,
    momentum_candidate_topk: int = 40,
    momentum_final_topk: int = 10,
    require_positive_ret_20d: bool = True,
    require_positive_ret_60d: bool = True,
    momentum_liquidity_quantile: float = 0.3,
    stock_clusters_df: pd.DataFrame | None = None,
    cluster_candidate_topk: int = 80,
    cluster_max_per_cluster: int = 3,
    cluster_require_positive_ret_20d: bool = False,
    cluster_require_positive_ret_60d: bool = False,
    cluster_require_positive_rsj_20d: bool = False,
    cluster_require_positive_rsj_60d: bool = False,
    daily_k_df: pd.DataFrame | None = None,
    execution_mode: str = "monthly_label",
    entry_delay_days: int = 1,
    stop_loss_pct: float = 0.08,
    trailing_stop_pct: float = 0.0,
    limit_threshold_pct: float = 9.8,
    slippage_bps: float = 5.0,
    max_monthly_turnover: float = 1.0,
    single_stock_max_weight: float = 0.10,
    max_hold_months: int = 0,
    entry_batch_days: int = 1,
    stop_loss_confirm_days: int = 1,
    entry_trigger: str = "immediate",
    entry_window_days: int = 15,
    min_pullback_pct: float = 0.03,
    max_pullback_pct: float = 0.08,
    breakout_lookback_days: int = 20,
    breakout_volume_multiplier: float = 1.2,
) -> pd.DataFrame:
    empty_columns = [
        "month",
        "selected_count",
        "portfolio_return",
        "portfolio_return_3m",
        "portfolio_return_6m",
        "benchmark_return",
        "benchmark_hs300_return",
        "benchmark_hs300_return_3m",
        "benchmark_hs300_return_6m",
        "benchmark_zz500_return",
        "benchmark_zz500_return_3m",
        "benchmark_zz500_return_6m",
        "excess_return",
        "excess_hs300_return",
        "excess_hs300_return_3m",
        "excess_hs300_return_6m",
        "excess_zz500_return",
        "excess_zz500_return_3m",
        "excess_zz500_return_6m",
        "turnover",
        "transaction_cost",
        "ic",
        "cumulative_net_value",
        "buy_codes",
        "sell_codes",
        "hold_codes",
    ]
    tradable = labels_df.drop(columns=["is_tradable"], errors="ignore").copy()
    scored = scores_df.merge(
        tradable,
        on=["month", "code"],
        how="inner",
    )
    if str(execution_mode).strip().lower() == "daily":
        if daily_k_df is None:
            raise ValueError("daily execution mode requires daily_k_df")
        monthly_df, _daily_returns, _events = run_daily_signal_backtest(
            scores_df=scored,
            daily_k_df=daily_k_df,
            benchmark_df=benchmark_df,
            monthly_features_df=tradability_df,
            trading_strategy=trading_strategy,
            cost_bps=cost_bps,
            k=k,
            buy_topk=buy_topk,
            hold_topk=hold_topk,
            max_positions=max_positions,
            liquidity_quantile=liquidity_quantile,
            min_price=min_price,
            exclude_st=exclude_st,
            momentum_candidate_topk=momentum_candidate_topk,
            momentum_final_topk=momentum_final_topk,
            require_positive_ret_20d=require_positive_ret_20d,
            require_positive_ret_60d=require_positive_ret_60d,
            momentum_liquidity_quantile=momentum_liquidity_quantile,
            stock_clusters_df=stock_clusters_df,
            cluster_candidate_topk=cluster_candidate_topk,
            cluster_max_per_cluster=cluster_max_per_cluster,
            cluster_require_positive_ret_20d=cluster_require_positive_ret_20d,
            cluster_require_positive_ret_60d=cluster_require_positive_ret_60d,
            cluster_require_positive_rsj_20d=cluster_require_positive_rsj_20d,
            cluster_require_positive_rsj_60d=cluster_require_positive_rsj_60d,
            entry_delay_days=entry_delay_days,
            stop_loss_pct=stop_loss_pct,
            trailing_stop_pct=trailing_stop_pct,
            limit_threshold_pct=limit_threshold_pct,
            slippage_bps=slippage_bps,
            max_monthly_turnover=max_monthly_turnover,
            single_stock_max_weight=single_stock_max_weight,
            max_hold_months=max_hold_months,
            entry_batch_days=entry_batch_days,
            stop_loss_confirm_days=stop_loss_confirm_days,
            entry_trigger=entry_trigger,
            entry_window_days=entry_window_days,
            min_pullback_pct=min_pullback_pct,
            max_pullback_pct=max_pullback_pct,
            breakout_lookback_days=breakout_lookback_days,
            breakout_volume_multiplier=breakout_volume_multiplier,
        )
        return monthly_df

    if scored.empty:
        return pd.DataFrame(columns=empty_columns)

    benchmark_indexed = benchmark_df.set_index("month") if "month" in benchmark_df.columns else pd.DataFrame()
    hs300_lookup: dict[int, dict[Any, Any]] = {}
    zz500_lookup: dict[int, dict[Any, Any]] = {}
    for horizon in FORWARD_RETURN_HORIZONS:
        hs300_column = (
            f"hs300_return_{horizon}m"
            if f"hs300_return_{horizon}m" in benchmark_df.columns
            else f"benchmark_return_{horizon}m"
        )
        zz500_column = f"zz500_return_{horizon}m"
        hs300_lookup[horizon] = (
            benchmark_indexed[hs300_column].to_dict()
            if not benchmark_indexed.empty and hs300_column in benchmark_indexed.columns
            else {}
        )
        zz500_lookup[horizon] = (
            benchmark_indexed[zz500_column].to_dict()
            if not benchmark_indexed.empty and zz500_column in benchmark_indexed.columns
            else {}
        )

    rows: list[dict[str, Any]] = []
    previous_codes: set[str] = set()
    cumulative_net_value = 1.0
    for month in sorted(scored["month"].unique()):
        month_scored = scored.loc[scored["month"] == month].copy()
        month_features = None
        if tradability_df is not None:
            month_features = tradability_df.loc[tradability_df["month"] == month].copy()
        month_rows = build_portfolio_by_strategy(
            trading_strategy=trading_strategy,
            scores_df=month_scored,
            monthly_features_df=month_features,
            previous_codes=previous_codes,
            k=k,
            buy_topk=buy_topk,
            hold_topk=hold_topk,
            max_positions=max_positions,
            liquidity_quantile=liquidity_quantile,
            min_price=min_price,
            exclude_st=exclude_st,
            momentum_candidate_topk=momentum_candidate_topk,
            momentum_final_topk=momentum_final_topk,
            require_positive_ret_20d=require_positive_ret_20d,
            require_positive_ret_60d=require_positive_ret_60d,
            momentum_liquidity_quantile=momentum_liquidity_quantile,
            stock_clusters_df=(
                stock_clusters_df.loc[stock_clusters_df["month"].astype(str) == str(month)].copy()
                if stock_clusters_df is not None and not stock_clusters_df.empty
                else None
            ),
            cluster_candidate_topk=cluster_candidate_topk,
            cluster_max_per_cluster=cluster_max_per_cluster,
            cluster_require_positive_ret_20d=cluster_require_positive_ret_20d,
            cluster_require_positive_ret_60d=cluster_require_positive_ret_60d,
            cluster_require_positive_rsj_20d=cluster_require_positive_rsj_20d,
            cluster_require_positive_rsj_60d=cluster_require_positive_rsj_60d,
        )
        current_codes = set(month_rows["code"])
        buy_codes = sorted(current_codes - previous_codes)
        sell_codes = sorted(previous_codes - current_codes)
        hold_codes = sorted(current_codes & previous_codes)
        turnover = _compute_turnover(previous_codes, current_codes)
        transaction_cost = turnover * cost_bps / 10000.0
        portfolio_returns: dict[int, float] = {}
        benchmark_hs300_returns: dict[int, float] = {}
        benchmark_zz500_returns: dict[int, float] = {}
        excess_hs300_returns: dict[int, float] = {}
        excess_zz500_returns: dict[int, float] = {}
        for horizon in FORWARD_RETURN_HORIZONS:
            stock_return_column = f"future_stock_return_{horizon}m"
            gross_return = (
                float(month_rows[stock_return_column].mean())
                if not month_rows.empty and stock_return_column in month_rows.columns
                else 0.0
            )
            portfolio_returns[horizon] = gross_return - transaction_cost
            benchmark_hs300_returns[horizon] = float(hs300_lookup[horizon].get(month, 0.0))
            benchmark_zz500_returns[horizon] = float(zz500_lookup[horizon].get(month, 0.0))
            excess_hs300_returns[horizon] = portfolio_returns[horizon] - benchmark_hs300_returns[horizon]
            excess_zz500_returns[horizon] = portfolio_returns[horizon] - benchmark_zz500_returns[horizon]
        portfolio_return = portfolio_returns[1]
        cumulative_net_value *= 1.0 + portfolio_return

        ic = _compute_ic(month_scored)
        rows.append(
            {
                "month": month,
                "selected_count": int(len(month_rows)),
                "portfolio_return": portfolio_return,
                "portfolio_return_3m": portfolio_returns[3],
                "portfolio_return_6m": portfolio_returns[6],
                "benchmark_return": benchmark_hs300_returns[1],
                "benchmark_hs300_return": benchmark_hs300_returns[1],
                "benchmark_hs300_return_3m": benchmark_hs300_returns[3],
                "benchmark_hs300_return_6m": benchmark_hs300_returns[6],
                "benchmark_zz500_return": benchmark_zz500_returns[1],
                "benchmark_zz500_return_3m": benchmark_zz500_returns[3],
                "benchmark_zz500_return_6m": benchmark_zz500_returns[6],
                "excess_return": excess_hs300_returns[1],
                "excess_hs300_return": excess_hs300_returns[1],
                "excess_hs300_return_3m": excess_hs300_returns[3],
                "excess_hs300_return_6m": excess_hs300_returns[6],
                "excess_zz500_return": excess_zz500_returns[1],
                "excess_zz500_return_3m": excess_zz500_returns[3],
                "excess_zz500_return_6m": excess_zz500_returns[6],
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "ic": ic,
                "cumulative_net_value": cumulative_net_value,
                "buy_codes": ",".join(buy_codes),
                "sell_codes": ",".join(sell_codes),
                "hold_codes": ",".join(hold_codes),
            }
        )
        previous_codes = current_codes

    return pd.DataFrame(rows, columns=empty_columns)


def compute_backtest_metrics(backtest_df: pd.DataFrame) -> dict[str, float | int]:
    if backtest_df.empty:
        return {
            "months": 0,
            "portfolio_total_return": 0.0,
            "benchmark_total_return": 0.0,
            "excess_total_return": 0.0,
            "benchmark_hs300_total_return": 0.0,
            "benchmark_zz500_total_return": 0.0,
            "excess_hs300_total_return": 0.0,
            "excess_zz500_total_return": 0.0,
            "average_turnover": 0.0,
            "mean_ic": 0.0,
            "sharpe_ratio": 0.0,
        }

    portfolio_total_return = float(backtest_df["cumulative_net_value"].iloc[-1] - 1.0)
    benchmark_hs300_total_return = float(
        (1.0 + backtest_df.get("benchmark_hs300_return", backtest_df["benchmark_return"]))
        .prod()
        - 1.0
    )
    benchmark_zz500_total_return = float(
        (1.0 + backtest_df.get("benchmark_zz500_return", pd.Series([0.0] * len(backtest_df))))
        .prod()
        - 1.0
    )
    excess_hs300_total_return = float(
        (1.0 + portfolio_total_return) / (1.0 + benchmark_hs300_total_return) - 1.0
    )
    excess_zz500_total_return = float(
        (1.0 + portfolio_total_return) / (1.0 + benchmark_zz500_total_return) - 1.0
    )
    sharpe_ratio = _compute_annualized_sharpe(backtest_df["portfolio_return"])
    return {
        "months": int(len(backtest_df)),
        "portfolio_total_return": portfolio_total_return,
        "benchmark_total_return": benchmark_hs300_total_return,
        "excess_total_return": excess_hs300_total_return,
        "benchmark_hs300_total_return": benchmark_hs300_total_return,
        "benchmark_zz500_total_return": benchmark_zz500_total_return,
        "excess_hs300_total_return": excess_hs300_total_return,
        "excess_zz500_total_return": excess_zz500_total_return,
        "average_turnover": float(backtest_df["turnover"].mean()),
        "mean_ic": float(backtest_df["ic"].mean()),
        "sharpe_ratio": sharpe_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run monthly alphagraph backtest")
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=ARTIFACT_DIR / "monthly_scores.parquet",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=ARTIFACT_DIR / "monthly_labels.parquet",
    )
    parser.add_argument(
        "--tradability-path",
        type=Path,
        default=None,
        help="Current-month tradability features. Defaults to monthly features next to scores.",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=ARTIFACT_DIR / "monthly_benchmark_returns.parquet",
    )
    parser.add_argument(
        "--output-returns",
        type=Path,
        default=ARTIFACT_DIR / "backtest_returns.parquet",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=ARTIFACT_DIR / "backtest_metrics.json",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-runs-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--stock-graph-edges-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--daily-k-path",
        type=Path,
        default=None,
        help="Daily OHLC data for K-line trade markers. Defaults to daily_k.parquet next to scores.",
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--cost-bps", type=float, default=20)
    parser.add_argument(
        "--trading-strategy",
        choices=("topk", "topk_buffer", "momentum_confirmed", "cluster_topk"),
        default="topk",
    )
    parser.add_argument("--buy-topk", type=int, default=20)
    parser.add_argument("--hold-topk", type=int, default=50)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--liquidity-quantile", type=float, default=0.2)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--exclude-st", type=_parse_bool, default=True)
    parser.add_argument("--momentum-candidate-topk", type=int, default=40)
    parser.add_argument("--momentum-final-topk", type=int, default=10)
    parser.add_argument("--require-positive-ret-20d", type=_parse_bool, default=True)
    parser.add_argument("--require-positive-ret-60d", type=_parse_bool, default=True)
    parser.add_argument("--momentum-liquidity-quantile", type=float, default=0.3)
    parser.add_argument("--stock-clusters-path", type=Path, default=None)
    parser.add_argument("--cluster-candidate-topk", type=int, default=80)
    parser.add_argument("--cluster-max-per-cluster", type=int, default=3)
    parser.add_argument("--cluster-require-positive-ret-20d", type=_parse_bool, default=False)
    parser.add_argument("--cluster-require-positive-ret-60d", type=_parse_bool, default=False)
    parser.add_argument("--cluster-require-positive-rsj-20d", type=_parse_bool, default=False)
    parser.add_argument("--cluster-require-positive-rsj-60d", type=_parse_bool, default=False)
    parser.add_argument("--execution-mode", choices=("monthly_label", "daily"), default="monthly_label")
    parser.add_argument("--entry-delay-days", type=int, default=1)
    parser.add_argument("--stop-loss-pct", type=float, default=0.08)
    parser.add_argument("--trailing-stop-pct", type=float, default=0.0)
    parser.add_argument("--limit-threshold-pct", type=float, default=9.8)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--max-monthly-turnover", type=float, default=1.0)
    parser.add_argument("--single-stock-max-weight", type=float, default=0.10)
    parser.add_argument("--max-hold-months", type=int, default=0)
    parser.add_argument("--entry-batch-days", type=int, default=1)
    parser.add_argument("--stop-loss-confirm-days", type=int, default=1)
    parser.add_argument("--entry-trigger", choices=("immediate", "pullback", "breakout", "pullback_or_breakout"), default="immediate")
    parser.add_argument("--entry-window-days", type=int, default=15)
    parser.add_argument("--min-pullback-pct", type=float, default=0.03)
    parser.add_argument("--max-pullback-pct", type=float, default=0.08)
    parser.add_argument("--breakout-lookback-days", type=int, default=20)
    parser.add_argument("--breakout-volume-multiplier", type=float, default=1.2)
    parser.add_argument("--config-name", type=str, default="")
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--experiment-description", type=str, default="")
    parser.add_argument("--fusion-type", type=str, default="")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--decay-months", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--training-label", type=str, default="")
    args = parser.parse_args()

    scores_df = _read_frame(args.scores_path)
    labels_df = _read_frame(args.labels_path)
    benchmark_df = _read_frame(args.benchmark_path)
    tradability_path = args.tradability_path or (args.scores_path.parent / "monthly_stock_features.parquet")
    tradability_df = _read_frame(tradability_path) if tradability_path.exists() else None
    stock_clusters_path = args.stock_clusters_path or (args.scores_path.parent / "stock_clusters.parquet")
    stock_clusters_df = _read_frame(stock_clusters_path) if stock_clusters_path.exists() else None
    daily_k_path = args.daily_k_path or (args.scores_path.parent / "daily_k.parquet")
    if args.execution_mode == "daily" and not daily_k_path.exists():
        raise FileNotFoundError(f"daily execution mode requires daily K data: {daily_k_path}")
    daily_k_df = _read_frame(daily_k_path) if args.execution_mode == "daily" and daily_k_path.exists() else None

    backtest_df = run_monthly_backtest(
        scores_df=scores_df,
        labels_df=labels_df,
        benchmark_df=benchmark_df,
        cost_bps=args.cost_bps,
        k=args.k,
        tradability_df=tradability_df,
        trading_strategy=args.trading_strategy,
        buy_topk=args.buy_topk,
        hold_topk=args.hold_topk,
        max_positions=args.max_positions,
        liquidity_quantile=args.liquidity_quantile,
        min_price=args.min_price,
        exclude_st=args.exclude_st,
        momentum_candidate_topk=args.momentum_candidate_topk,
        momentum_final_topk=args.momentum_final_topk,
        require_positive_ret_20d=args.require_positive_ret_20d,
        require_positive_ret_60d=args.require_positive_ret_60d,
        momentum_liquidity_quantile=args.momentum_liquidity_quantile,
        stock_clusters_df=stock_clusters_df,
        cluster_candidate_topk=args.cluster_candidate_topk,
        cluster_max_per_cluster=args.cluster_max_per_cluster,
        cluster_require_positive_ret_20d=args.cluster_require_positive_ret_20d,
        cluster_require_positive_ret_60d=args.cluster_require_positive_ret_60d,
        cluster_require_positive_rsj_20d=args.cluster_require_positive_rsj_20d,
        cluster_require_positive_rsj_60d=args.cluster_require_positive_rsj_60d,
        daily_k_df=daily_k_df,
        execution_mode=args.execution_mode,
        entry_delay_days=args.entry_delay_days,
        stop_loss_pct=args.stop_loss_pct,
        trailing_stop_pct=args.trailing_stop_pct,
        limit_threshold_pct=args.limit_threshold_pct,
        slippage_bps=args.slippage_bps,
        max_monthly_turnover=args.max_monthly_turnover,
        single_stock_max_weight=args.single_stock_max_weight,
        max_hold_months=args.max_hold_months,
        entry_batch_days=args.entry_batch_days,
        stop_loss_confirm_days=args.stop_loss_confirm_days,
        entry_trigger=args.entry_trigger,
        entry_window_days=args.entry_window_days,
        min_pullback_pct=args.min_pullback_pct,
        max_pullback_pct=args.max_pullback_pct,
        breakout_lookback_days=args.breakout_lookback_days,
        breakout_volume_multiplier=args.breakout_volume_multiplier,
    )
    metrics = compute_backtest_metrics(backtest_df)
    run_config = _build_run_config(
        config_name=args.config_name,
        fusion_type=args.fusion_type,
        model_name=args.model_name,
        decay_months=args.decay_months,
        epochs=args.epochs,
        training_label=args.training_label,
        topk=args.k,
        cost_bps=args.cost_bps,
        trading_strategy=args.trading_strategy,
        buy_topk=args.buy_topk,
        hold_topk=args.hold_topk,
        max_positions=args.max_positions,
        liquidity_quantile=args.liquidity_quantile,
        min_price=args.min_price,
        exclude_st=args.exclude_st,
        momentum_candidate_topk=args.momentum_candidate_topk,
        momentum_final_topk=args.momentum_final_topk,
        require_positive_ret_20d=args.require_positive_ret_20d,
        require_positive_ret_60d=args.require_positive_ret_60d,
        momentum_liquidity_quantile=args.momentum_liquidity_quantile,
        cluster_require_positive_ret_20d=args.cluster_require_positive_ret_20d,
        cluster_require_positive_ret_60d=args.cluster_require_positive_ret_60d,
        cluster_require_positive_rsj_20d=args.cluster_require_positive_rsj_20d,
        cluster_require_positive_rsj_60d=args.cluster_require_positive_rsj_60d,
        cluster_max_per_cluster=args.cluster_max_per_cluster,
        execution_mode=args.execution_mode,
        entry_delay_days=args.entry_delay_days,
        stop_loss_pct=args.stop_loss_pct,
        trailing_stop_pct=args.trailing_stop_pct,
        limit_threshold_pct=args.limit_threshold_pct,
        slippage_bps=args.slippage_bps,
        max_monthly_turnover=args.max_monthly_turnover,
        single_stock_max_weight=args.single_stock_max_weight,
        max_hold_months=args.max_hold_months,
        entry_batch_days=args.entry_batch_days,
        stop_loss_confirm_days=args.stop_loss_confirm_days,
        entry_trigger=args.entry_trigger,
        entry_window_days=args.entry_window_days,
        min_pullback_pct=args.min_pullback_pct,
        max_pullback_pct=args.max_pullback_pct,
        breakout_lookback_days=args.breakout_lookback_days,
        breakout_volume_multiplier=args.breakout_volume_multiplier,
        scores_path=args.scores_path,
        tradability_path=tradability_path,
        daily_k_path=daily_k_path,
        experiment_name=args.experiment_name,
        experiment_description=args.experiment_description,
    )
    metrics["run_config"] = run_config

    _write_frame(backtest_df, args.output_returns)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_html = args.output_html or _default_backtest_report_path(args.output_metrics.parent)
    model_runs_dir = _resolve_model_runs_dir(args.model_runs_dir, args.output_metrics.parent)
    stock_graph_edges_path = _resolve_stock_graph_edges_path(
        args.stock_graph_edges_path,
        args.scores_path,
    )
    daily_k_path = args.daily_k_path or (args.scores_path.parent / "daily_k.parquet")
    _write_html_report(
        backtest_df=backtest_df,
        metrics=metrics,
        output_path=output_html,
        scores_path=args.scores_path,
        labels_path=args.labels_path,
        benchmark_path=args.benchmark_path,
        output_returns_path=args.output_returns,
        output_metrics_path=args.output_metrics,
        model_runs_dir=model_runs_dir,
        stock_graph_edges_path=stock_graph_edges_path,
        daily_k_path=daily_k_path,
        run_config=run_config,
    )
    _write_backtest_index_html(ARTIFACT_DIR)


def _compute_turnover(previous_codes: set[str], current_codes: set[str]) -> float:
    if not current_codes:
        return 0.0
    if not previous_codes:
        return 1.0
    overlap = len(previous_codes & current_codes)
    return 1.0 - overlap / max(len(previous_codes), len(current_codes))


def _compute_ic(month_df: pd.DataFrame) -> float:
    if len(month_df) < 2:
        return 0.0
    correlation = month_df["score"].corr(month_df["future_stock_return_1m"])
    if pd.isna(correlation):
        return 0.0
    return float(correlation)


def _compute_annualized_sharpe(portfolio_returns: pd.Series, periods_per_year: int = 12) -> float:
    cleaned = portfolio_returns.dropna().astype(float)
    if len(cleaned) < 2:
        return 0.0
    volatility = float(cleaned.std(ddof=1))
    if volatility <= 0.0:
        return 0.0
    mean_return = float(cleaned.mean())
    return float(math.sqrt(periods_per_year) * mean_return / volatility)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _resolve_model_runs_dir(model_runs_dir: Path | None, output_dir: Path) -> Path:
    if model_runs_dir is not None:
        return model_runs_dir

    dataset_local_dir = output_dir / "model_runs"
    if dataset_local_dir.exists():
        return dataset_local_dir
    return ARTIFACT_DIR / "model_runs"


def _resolve_stock_graph_edges_path(
    stock_graph_edges_path: Path | None,
    scores_path: Path,
) -> Path:
    if stock_graph_edges_path is not None:
        return stock_graph_edges_path
    return scores_path.parent / "stock_graph_edges.parquet"


def _resolve_kg_stock_edges_llm_path(scores_path: Path) -> Path:
    return scores_path.parent / "kg_stock_edges_llm.parquet"


def _collect_model_runs(model_runs_dir: Path, max_runs: int = 10) -> list[dict[str, Any]]:
    if not model_runs_dir.exists() or not model_runs_dir.is_dir():
        return []

    run_dirs = sorted(
        [path for path in model_runs_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs[:max_runs]:
        metadata_path = run_dir / "run_metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        rows.append(
            {
                "run_id": str(metadata.get("run_id", run_dir.name)),
                "model_name": str(metadata.get("model_name", "")),
                "epochs": metadata.get("epochs", ""),
                "decay_months": metadata.get("decay_months", ""),
                "score_rows": metadata.get("score_rows", ""),
                "run_dir": str(run_dir),
                "metadata_path": str(metadata_path) if metadata_path.exists() else "",
            }
        )
    return rows


def _build_run_config(
    config_name: str,
    fusion_type: str,
    model_name: str,
    decay_months: int | None,
    epochs: int | None,
    training_label: str,
    topk: int,
    cost_bps: float,
    trading_strategy: str,
    buy_topk: int,
    hold_topk: int,
    max_positions: int,
    liquidity_quantile: float,
    min_price: float,
    exclude_st: bool,
    momentum_candidate_topk: int,
    momentum_final_topk: int,
    require_positive_ret_20d: bool,
    require_positive_ret_60d: bool,
    momentum_liquidity_quantile: float,
    cluster_require_positive_ret_20d: bool,
    cluster_require_positive_ret_60d: bool,
    cluster_require_positive_rsj_20d: bool,
    cluster_require_positive_rsj_60d: bool,
    cluster_max_per_cluster: int,
    execution_mode: str,
    entry_delay_days: int,
    stop_loss_pct: float,
    trailing_stop_pct: float,
    limit_threshold_pct: float,
    slippage_bps: float,
    max_monthly_turnover: float,
    single_stock_max_weight: float,
    max_hold_months: int,
    entry_batch_days: int,
    stop_loss_confirm_days: int,
    entry_trigger: str,
    entry_window_days: int,
    min_pullback_pct: float,
    max_pullback_pct: float,
    breakout_lookback_days: int,
    breakout_volume_multiplier: float,
    scores_path: Path,
    tradability_path: Path | None = None,
    daily_k_path: Path | None = None,
    experiment_name: str = "",
    experiment_description: str = "",
) -> dict[str, Any]:
    inferred_fusion_type = fusion_type.strip() or _infer_fusion_type_from_scores_path(scores_path)
    inferred_config_name = config_name.strip() or (
        f"fusion_{inferred_fusion_type}" if inferred_fusion_type else ""
    )
    resolved_experiment_name = experiment_name.strip() or inferred_config_name
    return {
        "config_name": inferred_config_name,
        "experiment_name": resolved_experiment_name,
        "experiment_description": experiment_description.strip(),
        "fusion_type": inferred_fusion_type,
        "model_name": model_name.strip(),
        "decay_months": decay_months if decay_months is not None else "",
        "epochs": epochs if epochs is not None else "",
        "training_label": training_label.strip(),
        "topk": int(topk),
        "cost_bps": float(cost_bps),
        "trading_strategy": trading_strategy,
        "buy_topk": int(buy_topk),
        "hold_topk": int(hold_topk),
        "max_positions": int(max_positions),
        "liquidity_quantile": float(liquidity_quantile),
        "min_price": float(min_price),
        "exclude_st": bool(exclude_st),
        "momentum_candidate_topk": int(momentum_candidate_topk),
        "momentum_final_topk": int(momentum_final_topk),
        "require_positive_ret_20d": bool(require_positive_ret_20d),
        "require_positive_ret_60d": bool(require_positive_ret_60d),
        "momentum_liquidity_quantile": float(momentum_liquidity_quantile),
        "cluster_require_positive_ret_20d": bool(cluster_require_positive_ret_20d),
        "cluster_require_positive_ret_60d": bool(cluster_require_positive_ret_60d),
        "cluster_require_positive_rsj_20d": bool(
            cluster_require_positive_rsj_20d or cluster_require_positive_ret_20d
        ),
        "cluster_require_positive_rsj_60d": bool(
            cluster_require_positive_rsj_60d or cluster_require_positive_ret_60d
        ),
        "cluster_max_per_cluster": int(cluster_max_per_cluster),
        "execution_mode": execution_mode,
        "entry_delay_days": int(entry_delay_days),
        "stop_loss_pct": float(stop_loss_pct),
        "trailing_stop_pct": float(trailing_stop_pct),
        "limit_threshold_pct": float(limit_threshold_pct),
        "slippage_bps": float(slippage_bps),
        "max_monthly_turnover": float(max_monthly_turnover),
        "single_stock_max_weight": float(single_stock_max_weight),
        "max_hold_months": int(max_hold_months),
        "entry_batch_days": int(entry_batch_days),
        "stop_loss_confirm_days": int(stop_loss_confirm_days),
        "entry_trigger": entry_trigger,
        "entry_window_days": int(entry_window_days),
        "min_pullback_pct": float(min_pullback_pct),
        "max_pullback_pct": float(max_pullback_pct),
        "breakout_lookback_days": int(breakout_lookback_days),
        "breakout_volume_multiplier": float(breakout_volume_multiplier),
        "scores_path": str(scores_path),
        "tradability_filter": "current_month",
        "tradability_path": str(tradability_path) if tradability_path is not None else "",
        "daily_k_path": str(daily_k_path) if daily_k_path is not None else "",
    }


def _infer_fusion_type_from_scores_path(scores_path: Path) -> str:
    stem = scores_path.stem
    prefix = "monthly_scores_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return ""


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "-"


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _format_config_value(value: Any) -> str:
    if value is None or value == "":
        return "未记录"
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


def _normalize_report_run_config(run_config: dict[str, Any], scores_path: Path) -> dict[str, Any]:
    normalized = dict(run_config)
    inferred_fusion_type = normalized.get("fusion_type") or _infer_fusion_type_from_scores_path(scores_path)
    if inferred_fusion_type:
        normalized["fusion_type"] = inferred_fusion_type
    normalized.setdefault(
        "config_name",
        f"fusion_{inferred_fusion_type}" if inferred_fusion_type else "",
    )
    normalized.setdefault("scores_path", str(scores_path))
    return normalized


def _build_backtest_diagnostics(
    metrics: dict[str, Any],
    run_config: dict[str, Any],
    backtest_df: pd.DataFrame,
    scores_path: Path,
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    months = int(metrics.get("months", 0) or 0)
    single_month_mode = str(run_config.get("single_month_mode", "")).strip().lower()
    if months <= 0:
        diagnostics.append(
            {
                "title": "月份不足无法训练回测",
                "description": "当前输入只形成了 0 个有效回测月份，系统不会伪造收益数据。",
            }
        )
    if single_month_mode == "diagnostic":
        diagnostics.append(
            {
                "title": "单月诊断模式",
                "description": "只有 1 个有效月份，已降级为诊断输出，不执行同月训练同月评估。",
            }
        )
    if "month" not in backtest_df.columns:
        diagnostics.append(
            {
                "title": "月度结果缺失",
                "description": f"{scores_path.name} 未产出 month 列，因此只展示诊断信息。",
            }
        )
    return diagnostics


def _format_chart_time(month_value: Any) -> str:
    raw = str(month_value).strip()
    if len(raw) == 7 and raw[4] == "-":
        return f"{raw}-01"
    return raw


def _build_chart_series(backtest_df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if column not in backtest_df.columns or "month" not in backtest_df.columns:
        return []
    series: list[dict[str, Any]] = []
    ordered = backtest_df.sort_values("month")
    for row in ordered.itertuples(index=False):
        value = getattr(row, column, None)
        month = getattr(row, "month", "")
        if pd.isna(value) or pd.isna(month):
            continue
        series.append(
            {
                "time": _format_chart_time(month),
                "value": float(value),
            }
        )
    return series


def _build_gnn_graph_payload(
    scores_path: Path,
    stock_graph_edges_path: Path,
    max_nodes: int = 30,
    max_edges: int = 120,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "month": "",
        "nodes": [],
        "links": [],
        "stock_graph_edges_path": str(stock_graph_edges_path),
    }
    if not scores_path.exists() or not stock_graph_edges_path.exists():
        return payload

    try:
        scores_df = _read_frame(scores_path)
    except Exception:
        return payload

    if scores_df.empty or not {"month", "code", "score"}.issubset(scores_df.columns):
        return payload

    valid_scores = scores_df.dropna(subset=["month", "code", "score"]).copy()
    if valid_scores.empty:
        return payload

    latest_month = sorted(valid_scores["month"].astype(str).unique())[-1]
    month_scores = valid_scores.loc[valid_scores["month"].astype(str) == latest_month].copy()
    month_scores = month_scores.sort_values(["score", "code"], ascending=[False, True]).reset_index(drop=True)
    top_nodes_df = month_scores.head(max_nodes).copy()
    if top_nodes_df.empty:
        payload["month"] = latest_month
        return payload

    top_nodes_df["rank"] = top_nodes_df.index + 1
    top_node_codes = set(top_nodes_df["code"].astype(str).tolist())

    nodes = [
        {
            "id": str(row.code),
            "label": str(row.code),
            "score": float(row.score),
            "rank": int(row.rank),
        }
        for row in top_nodes_df.itertuples(index=False)
    ]

    try:
        edges_df = _read_frame(stock_graph_edges_path)
    except Exception:
        payload["month"] = latest_month
        payload["nodes"] = nodes
        return payload

    links: list[dict[str, Any]] = []
    if (
        not edges_df.empty
        and "src_code" in edges_df.columns
        and "dst_code" in edges_df.columns
    ):
        filtered = edges_df.copy()
        filtered["src_code"] = filtered["src_code"].astype(str)
        filtered["dst_code"] = filtered["dst_code"].astype(str)
        filtered = filtered.loc[
            filtered["src_code"].isin(top_node_codes)
            & filtered["dst_code"].isin(top_node_codes)
            & (filtered["src_code"] != filtered["dst_code"])
        ].copy()

        if "edge_weight" in filtered.columns:
            filtered["edge_weight"] = pd.to_numeric(filtered["edge_weight"], errors="coerce").fillna(0.0)
            filtered = filtered.sort_values("edge_weight", ascending=False)
        filtered = filtered.head(max_edges)

        for row in filtered.itertuples(index=False):
            links.append(
                {
                    "source": str(row.src_code),
                    "target": str(row.dst_code),
                    "edge_type": str(getattr(row, "edge_type", "unknown")),
                    "edge_weight": float(getattr(row, "edge_weight", 0.0) or 0.0),
                }
            )

    payload["month"] = latest_month
    payload["nodes"] = nodes
    payload["links"] = links
    return payload


def _build_kg_graph_payload(kg_stock_edges_llm_path: Path) -> dict[str, Any]:
    return _shared_build_kg_graph_payload(kg_stock_edges_llm_path=kg_stock_edges_llm_path)


def _split_codes(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [
        code.strip()
        for code in str(value).split(",")
        if code.strip()
    ]


def _month_end_date(month: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(f"{month}-01", errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed + pd.offsets.MonthEnd(0)


def _build_trade_signal_rows(backtest_df: pd.DataFrame) -> list[dict[str, str]]:
    if backtest_df.empty or "month" not in backtest_df.columns:
        return []

    rows: list[dict[str, str]] = []
    for row in backtest_df.sort_values("month").itertuples(index=False):
        month = str(getattr(row, "month", ""))
        buy_codes = _split_codes(getattr(row, "buy_codes", ""))
        sell_codes = _split_codes(getattr(row, "sell_codes", ""))
        hold_codes = _split_codes(getattr(row, "hold_codes", ""))
        rows.append(
            {
                "month": month,
                "buy_codes": ", ".join(buy_codes) if buy_codes else "-",
                "sell_codes": ", ".join(sell_codes) if sell_codes else "-",
                "hold_codes": ", ".join(hold_codes) if hold_codes else "-",
                "buy_count": str(len(buy_codes)),
                "sell_count": str(len(sell_codes)),
                "hold_count": str(len(hold_codes)),
            }
        )
    return rows


def _parse_trade_event_values(value: Any, action_label: str, month: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for item in str(value or "").split(";"):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        events.append(
            {
                "month": month,
                "time": parts[0],
                "code": parts[1],
                "action": action_label,
            }
        )
    return events


def _build_kline_trade_payload(
    backtest_df: pd.DataFrame,
    daily_k_path: Path | None,
    max_codes: int = 12,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "daily_k_path": str(daily_k_path) if daily_k_path is not None else "",
        "codes": [],
        "series_by_code": {},
    }
    if daily_k_path is None or not daily_k_path.exists():
        return payload
    if backtest_df.empty or "month" not in backtest_df.columns:
        return payload

    signal_events: dict[str, list[dict[str, str]]] = {}
    ordered_codes: list[str] = []
    for row in backtest_df.sort_values("month").itertuples(index=False):
        month = str(getattr(row, "month", ""))
        parsed_events = []
        parsed_events.extend(_parse_trade_event_values(getattr(row, "buy_events", ""), "买入", month))
        parsed_events.extend(_parse_trade_event_values(getattr(row, "sell_events", ""), "卖出", month))
        if not parsed_events:
            marker_date = _month_end_date(month)
            if marker_date is None:
                continue
            marker_time = marker_date.strftime("%Y-%m-%d")
            for action_column, action_label in (("buy_codes", "买入"), ("sell_codes", "卖出")):
                for code in _split_codes(getattr(row, action_column, "")):
                    parsed_events.append(
                        {
                            "month": month,
                            "time": marker_time,
                            "code": code,
                            "action": action_label,
                        }
                    )
        for event in parsed_events:
            code = event["code"]
            if code not in signal_events:
                signal_events[code] = []
                ordered_codes.append(code)
            signal_events[code].append(event)

    if not ordered_codes:
        return payload

    selected_codes = ordered_codes[:max_codes]
    try:
        daily_df = _read_frame(daily_k_path)
    except Exception:
        return payload
    required_columns = {"date", "code", "open", "high", "low", "close"}
    if daily_df.empty or not required_columns.issubset(daily_df.columns):
        return payload

    daily = daily_df.loc[daily_df["code"].astype(str).isin(selected_codes)].copy()
    if daily.empty:
        return payload
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    for column in ["open", "high", "low", "close"]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily = daily.dropna(subset=["date", "code", "open", "high", "low", "close"])
    if daily.empty:
        return payload
    daily["code"] = daily["code"].astype(str)
    daily["time"] = daily["date"].dt.strftime("%Y-%m-%d")

    series_by_code: dict[str, dict[str, Any]] = {}
    for code in selected_codes:
        code_daily = daily.loc[daily["code"] == code].sort_values("date").copy()
        if code_daily.empty:
            continue
        candles = [
            {
                "time": str(row.time),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
            }
            for row in code_daily.itertuples(index=False)
        ]
        available_times = set(code_daily["time"].tolist())
        markers = []
        for event in signal_events.get(code, []):
            marker_time = event["time"]
            if marker_time not in available_times:
                visible = code_daily.loc[code_daily["date"] <= pd.to_datetime(marker_time)]
                if visible.empty:
                    visible = code_daily
                marker_time = str(visible.iloc[-1]["time"])
            is_buy = event["action"] == "买入"
            markers.append(
                {
                    "time": marker_time,
                    "position": "belowBar" if is_buy else "aboveBar",
                    "color": "#0f766e" if is_buy else "#be123c",
                    "shape": "arrowUp" if is_buy else "arrowDown",
                    "text": f"{event['month']} {event['action']}",
                }
            )
        markers = sorted(markers, key=lambda marker: (marker["time"], marker["text"]))
        series_by_code[code] = {
            "candles": candles,
            "markers": markers,
        }

    payload["codes"] = list(series_by_code.keys())
    payload["series_by_code"] = series_by_code
    return payload


def _write_backtest_index_html(report_root: Path, output_path: Path | None = None) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    output_html = output_path or (report_root / "index.html")
    rows = _collect_backtest_index_rows(report_root)
    month_options = sorted({row["month"] for row in rows}, reverse=True)
    rows_json = json.dumps(rows, ensure_ascii=False)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>超图回测报告索引</title>
    <link rel=\"stylesheet\" href=\"https://unpkg.com/antd@5/dist/reset.css\" />
    <style>
        :root {{ --page-bg: #f5f7fb; --text-muted: #667085; --panel-border: #e6ebf0; }}
        * {{ box-sizing: border-box; }}
        html, body {{
            width: 100%;
            max-width: 100%;
            overflow-x: hidden;
        }}
        body {{
            margin: 0;
            background: var(--page-bg);
            color: #172033;
            font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", \"PingFang SC\", \"Noto Sans SC\", sans-serif;
        }}
        #root {{ min-height: 100vh; }}
        .report-shell {{
            width: 100%;
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px clamp(8px, 2vw, 16px) 48px;
            min-width: 0;
        }}
        .report-stack {{ display: grid; gap: 16px; min-width: 0; }}
        .chart {{ width: 100%; height: 360px; border: 1px solid var(--panel-border); border-radius: 8px; }}
        .table-wrap {{ width: 100%; overflow-x: auto; }}
        .mono-cell {{ word-break: break-all; }}
        .muted-note {{ color: var(--text-muted); font-size: 13px; }}
        .chart-warning {{ display: none; margin-bottom: 10px; }}
        .ant-card, .ant-card-body {{
            border-radius: 8px;
            max-width: 100%;
            min-width: 0;
        }}
        .ant-table-wrapper, .ant-table, .ant-table-container {{
            max-width: 100%;
            min-width: 0;
        }}
        .ant-table-cell, .ant-descriptions-item-content {{
            overflow-wrap: anywhere;
            word-break: break-word;
        }}
        .ant-descriptions, .ant-descriptions-view, .ant-descriptions table {{
            max-width: 100%;
            table-layout: fixed;
        }}
        @media (max-width: 900px) {{ .chart {{ height: 280px; }} }}
    </style>
</head>
<body>
    <div id=\"root\"></div>
    <script src=\"https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js\"></script>
    <script crossorigin src=\"https://unpkg.com/react@18/umd/react.production.min.js\"></script>
    <script crossorigin src=\"https://unpkg.com/react-dom@18/umd/react-dom.production.min.js\"></script>
    <script src=\"https://unpkg.com/dayjs@1/dayjs.min.js\"></script>
    <script crossorigin src=\"https://unpkg.com/antd@5/dist/antd.min.js\"></script>
    <script>
        const rows = {rows_json};
        const generatedAt = {json.dumps(generated_at, ensure_ascii=False)};
        const monthOptions = {json.dumps(month_options, ensure_ascii=False)};
        const {{ createElement: h }} = React;
        const {{ ConfigProvider, Alert, Card, Descriptions, Select, Space, Table, Tag, Typography }} = antd;
        const {{ Title, Paragraph, Text }} = Typography;
        let netValueChart = null;
        const palette = ['#0f766e', '#1d4ed8', '#ea580c', '#be123c', '#7c3aed', '#0891b2', '#4d7c0f'];
        const addLineSeriesCompat = (chart, chartLib, options) => {{
            if (typeof chart.addLineSeries === 'function') return chart.addLineSeries(options);
            return chart.addSeries(chartLib.LineSeries, options);
        }};
        const renderNetValueComparison = (visible) => {{
            const container = document.getElementById('chart-net-value-compare');
            const warning = document.getElementById('chart-warning');
            if (!container) return;
            container.innerHTML = '';
            netValueChart = null;
            if (!window.LightweightCharts || typeof window.LightweightCharts.createChart !== 'function') {{
                if (warning) warning.style.display = 'block';
                return;
            }}
            if (warning) warning.style.display = 'none';
            const chartLib = window.LightweightCharts;
            const chart = chartLib.createChart(container, {{
                width: container.clientWidth,
                height: container.clientHeight,
                layout: {{ background: {{ color: '#ffffff' }}, textColor: '#334155' }},
                grid: {{ vertLines: {{ color: '#edf2f7' }}, horzLines: {{ color: '#edf2f7' }} }},
                rightPriceScale: {{ borderColor: '#cbd5e1' }},
                timeScale: {{ borderColor: '#cbd5e1' }},
            }});
            visible.forEach((row, index) => {{
                if (!Array.isArray(row.net_value_series) || row.net_value_series.length === 0) return;
                const label = `${{row.month}} ${{row.run_id}} ${{row.experiment_name || row.config_name || row.fusion_type || ''}}`.trim();
                const series = addLineSeriesCompat(chart, chartLib, {{ color: palette[index % palette.length], lineWidth: 2, title: label }});
                series.setData(row.net_value_series);
            }});
            chart.timeScale().fitContent();
            netValueChart = {{ chart, container }};
        }};
        function App() {{
            const [selectedMonth, setSelectedMonth] = React.useState('all');
            const visible = React.useMemo(() => selectedMonth === 'all' ? rows : rows.filter((row) => row.month === selectedMonth), [selectedMonth]);
            React.useEffect(() => {{ renderNetValueComparison(visible); }}, [visible]);
            React.useEffect(() => {{
                const handleResize = () => {{
                    if (netValueChart) {{
                        netValueChart.chart.applyOptions({{ width: netValueChart.container.clientWidth }});
                        netValueChart.chart.timeScale().fitContent();
                    }}
                }};
                window.addEventListener('resize', handleResize);
                return () => window.removeEventListener('resize', handleResize);
            }}, []);
            return h(ConfigProvider, {{ theme: {{ token: {{ colorPrimary: '#1677ff', borderRadius: 8 }} }} }},
                h('main', {{ className: 'report-shell' }},
                    h('div', {{ className: 'report-stack' }},
                        h(Card, null,
                            h(Space, {{ direction: 'vertical', size: 8 }},
                                h(Title, {{ level: 2, style: {{ margin: 0 }} }}, '超图回测报告索引'),
                                h(Space, {{ wrap: true }}, h(Tag, {{ color: 'blue' }}, `生成时间: ${{generatedAt}}`), h(Tag, null, 'Lightweight Charts'), h(Tag, null, 'Ant Design')),
                                h(Paragraph, {{ className: 'muted-note', style: {{ margin: 0 }} }}, '新增报告后可重新生成本页，或手动维护内嵌数据。'),
                            ),
                        ),
                        h(Card, null,
                            h('div', {{ style: {{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }} }},
                                h(Space, {{ align: 'center' }}, h(Text, null, '月份选择'), h(Select, {{ value: selectedMonth, style: {{ width: 180 }}, onChange: setSelectedMonth, options: [{{ value: 'all', label: '全部月份' }}, ...monthOptions.map((month) => ({{ value: month, label: month }}))] }})),
                                h(Tag, {{ color: 'blue' }}, `当前显示 ${{visible.length}} / ${{rows.length}} 个报告`),
                            ),
                        ),
                        h(Card, {{ title: '累计净值折线对比' }}, h(Alert, {{ id: 'chart-warning', className: 'chart-warning', type: 'warning', showIcon: true, title: '未加载 Lightweight Charts，无法显示折线图。' }}), h('div', {{ id: 'chart-net-value-compare', className: 'chart' }})),
                        h(Card, {{ title: '报告对比' }}, h(Table, {{ rowKey: (row) => `${{row.dataset}}-${{row.run_id}}-${{row.month}}`, size: 'small', dataSource: visible, columns: [
                            {{ title: '月份', dataIndex: 'month', fixed: 'left', width: 96 }},
                            {{ title: '数据集', dataIndex: 'dataset', width: 220 }},
                            {{ title: 'run', dataIndex: 'run_id', width: 180 }},
                            {{ title: '配置', dataIndex: 'config_name', width: 150, render: (value) => value || '-' }},
                            {{ title: '实验项目', dataIndex: 'experiment_name', width: 180, render: (value) => value || '-' }},
                            {{ title: '实验说明', dataIndex: 'experiment_description', width: 260, render: (value) => value || '-' }},
                            {{ title: '融合模块', dataIndex: 'fusion_type', width: 120, render: (value) => value || '-' }},
                            {{ title: '策略', dataIndex: 'trading_strategy', width: 140, render: (value) => value || '-' }},
                            {{ title: '组合总收益', dataIndex: 'portfolio_total_return', width: 120, render: (value) => Number.isFinite(Number(value)) ? `${{(Number(value) * 100).toFixed(2)}}%` : '-' }},
                            {{ title: '沪深300', dataIndex: 'benchmark_hs300_total_return', width: 110, render: (value) => Number.isFinite(Number(value)) ? `${{(Number(value) * 100).toFixed(2)}}%` : '-' }},
                            {{ title: '中证500', dataIndex: 'benchmark_zz500_total_return', width: 110, render: (value) => Number.isFinite(Number(value)) ? `${{(Number(value) * 100).toFixed(2)}}%` : '-' }},
                            {{ title: '超额HS300', dataIndex: 'excess_hs300_total_return', width: 120, render: (value) => Number.isFinite(Number(value)) ? `${{(Number(value) * 100).toFixed(2)}}%` : '-' }},
                            {{ title: '超额ZZ500', dataIndex: 'excess_zz500_total_return', width: 120, render: (value) => Number.isFinite(Number(value)) ? `${{(Number(value) * 100).toFixed(2)}}%` : '-' }},
                            {{ title: '平均IC', dataIndex: 'mean_ic', width: 96, render: (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(4) : '-' }},
                            {{ title: 'Sharpe', dataIndex: 'sharpe_ratio', width: 96, render: (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(4) : '-' }},
                            {{ title: '平均换手', dataIndex: 'average_turnover', width: 100, render: (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(4) : '-' }},
                            {{ title: '报告', dataIndex: 'report_href', fixed: 'right', width: 88, render: (href) => h('a', {{ href }}, '打开') }},
                        ], scroll: {{ x: 'max-content' }}, pagination: {{ pageSize: 20, showSizeChanger: true }}, locale: {{ emptyText: '没有匹配的报告。' }} }})),
                    ),
                ),
            );
        }}
        ReactDOM.createRoot(document.getElementById('root')).render(h(App));
    </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
    return output_html


def _collect_backtest_index_rows(report_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(report_root.glob("*/backtest_runs/*/backtest_metrics.json")):
        run_dir = metrics_path.parent
        dataset_dir = run_dir.parents[1]
        report_path = _find_backtest_report_path(run_dir)
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
        run_config = metrics.get("run_config") if isinstance(metrics.get("run_config"), dict) else {}
        dataset = dataset_dir.name
        rows.append(
            {
                "month": _extract_dataset_month(dataset),
                "dataset": dataset,
                "run_id": run_dir.name,
                "config_name": run_config.get("config_name", _infer_config_name_from_run_id(run_dir.name)),
                "experiment_name": run_config.get(
                    "experiment_name",
                    run_config.get("config_name", _infer_config_name_from_run_id(run_dir.name)),
                ),
                "experiment_description": run_config.get("experiment_description", ""),
                "fusion_type": run_config.get("fusion_type", ""),
                "trading_strategy": run_config.get("trading_strategy", "topk"),
                "portfolio_total_return": metrics.get("portfolio_total_return", ""),
                "benchmark_hs300_total_return": metrics.get(
                    "benchmark_hs300_total_return",
                    metrics.get("benchmark_total_return", ""),
                ),
                "benchmark_zz500_total_return": metrics.get("benchmark_zz500_total_return", ""),
                "excess_hs300_total_return": metrics.get(
                    "excess_hs300_total_return",
                    metrics.get("excess_total_return", ""),
                ),
                "excess_zz500_total_return": metrics.get("excess_zz500_total_return", ""),
                "average_turnover": metrics.get("average_turnover", ""),
                "mean_ic": metrics.get("mean_ic", ""),
                "sharpe_ratio": metrics.get("sharpe_ratio", ""),
                "report_href": report_path.relative_to(report_root).as_posix(),
                "net_value_series": _load_index_net_value_series(run_dir / "backtest_returns.parquet"),
            }
        )
    return sorted(rows, key=lambda row: (row["month"], row["run_id"]), reverse=True)


def _load_index_net_value_series(returns_path: Path) -> list[dict[str, Any]]:
    if not returns_path.exists():
        return []
    try:
        returns_df = _read_frame(returns_path)
    except Exception:
        return []
    return _build_chart_series(returns_df, "cumulative_net_value")


def _extract_dataset_month(dataset: str) -> str:
    prefix = dataset.split("_", 1)[0]
    return prefix if prefix.isdigit() else dataset


def _find_backtest_report_path(run_dir: Path) -> Path:
    named_report = _default_backtest_report_path(run_dir)
    if named_report.exists():
        return named_report
    reports = sorted(run_dir.glob("*.html"))
    if reports:
        return reports[0]
    return run_dir / "backtest_report.html"


def _infer_config_name_from_run_id(run_id: str) -> str:
    marker = "_fusion_"
    if marker in run_id:
        return f"fusion_{run_id.split(marker, 1)[1]}"
    return ""


def _write_html_report(
    backtest_df: pd.DataFrame,
    metrics: dict[str, Any],
    output_path: Path,
    scores_path: Path,
    labels_path: Path,
    benchmark_path: Path,
    output_returns_path: Path,
    output_metrics_path: Path,
    model_runs_dir: Path,
    stock_graph_edges_path: Path | None = None,
    daily_k_path: Path | None = None,
    run_config: dict[str, Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_runs = _collect_model_runs(model_runs_dir)
    resolved_stock_graph_edges_path = _resolve_stock_graph_edges_path(
        stock_graph_edges_path,
        scores_path,
    )
    resolved_kg_stock_edges_llm_path = _resolve_kg_stock_edges_llm_path(scores_path)
    resolved_daily_k_path = daily_k_path or (scores_path.parent / "daily_k.parquet")
    effective_run_config = _normalize_report_run_config(
        run_config or dict(metrics.get("run_config") or {}),
        scores_path,
    )

    metric_rows = [
        ("回测月份数", str(metrics.get("months", 0))),
        ("组合总收益", _format_pct(metrics.get("portfolio_total_return", 0.0))),
        ("沪深300总收益", _format_pct(metrics.get("benchmark_hs300_total_return", 0.0))),
        ("中证500总收益", _format_pct(metrics.get("benchmark_zz500_total_return", 0.0))),
        ("相对沪深300超额", _format_pct(metrics.get("excess_hs300_total_return", 0.0))),
        ("相对中证500超额", _format_pct(metrics.get("excess_zz500_total_return", 0.0))),
        ("平均换手率", _format_float(metrics.get("average_turnover", 0.0))),
        ("平均IC", _format_float(metrics.get("mean_ic", 0.0))),
        ("夏普比率(年化)", _format_float(metrics.get("sharpe_ratio", 0.0))),
    ]
    metric_table_rows = "\n".join(
        f"<tr><th>{escape(name)}</th><td>{escape(value)}</td></tr>" for name, value in metric_rows
    )
    config_rows = [
        ("CONFIG_NAME", effective_run_config.get("config_name", "")),
        ("实验项目", effective_run_config.get("experiment_name", "")),
        ("实验说明", effective_run_config.get("experiment_description", "")),
        ("FUSION_TYPE", effective_run_config.get("fusion_type", "")),
        ("TRADING_STRATEGY", effective_run_config.get("trading_strategy", "topk")),
        ("TOPK", effective_run_config.get("topk", "")),
        ("BUY_TOPK", effective_run_config.get("buy_topk", "")),
        ("HOLD_TOPK", effective_run_config.get("hold_topk", "")),
        ("MAX_POSITIONS", effective_run_config.get("max_positions", "")),
        ("LIQUIDITY_QUANTILE", effective_run_config.get("liquidity_quantile", "")),
        ("MIN_PRICE", effective_run_config.get("min_price", "")),
        ("EXCLUDE_ST", effective_run_config.get("exclude_st", "")),
        ("MOMENTUM_CANDIDATE_TOPK", effective_run_config.get("momentum_candidate_topk", "")),
        ("MOMENTUM_FINAL_TOPK", effective_run_config.get("momentum_final_topk", "")),
        ("REQUIRE_POSITIVE_RET_20D", effective_run_config.get("require_positive_ret_20d", "")),
        ("REQUIRE_POSITIVE_RET_60D", effective_run_config.get("require_positive_ret_60d", "")),
        ("MOMENTUM_LIQUIDITY_QUANTILE", effective_run_config.get("momentum_liquidity_quantile", "")),
        ("EXECUTION_MODE", effective_run_config.get("execution_mode", "")),
        ("ENTRY_DELAY_DAYS", effective_run_config.get("entry_delay_days", "")),
        ("STOP_LOSS_PCT", effective_run_config.get("stop_loss_pct", "")),
        ("TRAILING_STOP_PCT", effective_run_config.get("trailing_stop_pct", "")),
        ("LIMIT_THRESHOLD_PCT", effective_run_config.get("limit_threshold_pct", "")),
        ("SLIPPAGE_BPS", effective_run_config.get("slippage_bps", "")),
        ("MAX_MONTHLY_TURNOVER", effective_run_config.get("max_monthly_turnover", "")),
        ("MAX_HOLD_MONTHS", effective_run_config.get("max_hold_months", "")),
        ("ENTRY_BATCH_DAYS", effective_run_config.get("entry_batch_days", "")),
        ("STOP_LOSS_CONFIRM_DAYS", effective_run_config.get("stop_loss_confirm_days", "")),
        ("ENTRY_TRIGGER", effective_run_config.get("entry_trigger", "")),
        ("ENTRY_WINDOW_DAYS", effective_run_config.get("entry_window_days", "")),
        ("MIN_PULLBACK_PCT", effective_run_config.get("min_pullback_pct", "")),
        ("MAX_PULLBACK_PCT", effective_run_config.get("max_pullback_pct", "")),
        ("BREAKOUT_LOOKBACK_DAYS", effective_run_config.get("breakout_lookback_days", "")),
        ("BREAKOUT_VOLUME_MULTIPLIER", effective_run_config.get("breakout_volume_multiplier", "")),
        ("COST_BPS", effective_run_config.get("cost_bps", "")),
        ("DECAY_MONTHS", effective_run_config.get("decay_months", "")),
        ("EPOCHS", effective_run_config.get("epochs", "")),
        ("TRAINING_LABEL", effective_run_config.get("training_label", "")),
        ("MODEL_NAME", effective_run_config.get("model_name", "")),
        ("SCORES_PATH", effective_run_config.get("scores_path", str(scores_path))),
        ("TRADABILITY_FILTER", effective_run_config.get("tradability_filter", "")),
        ("TRADABILITY_PATH", effective_run_config.get("tradability_path", "")),
        ("DAILY_K_PATH", effective_run_config.get("daily_k_path", "")),
    ]
    config_table_rows = "\n".join(
        f"<tr><th>{escape(name)}</th><td>{escape(_format_config_value(value))}</td></tr>"
        for name, value in config_rows
    )

    monthly_columns = [
        "month",
        "selected_count",
        "portfolio_return",
        "portfolio_return_3m",
        "portfolio_return_6m",
        "benchmark_hs300_return",
        "benchmark_hs300_return_3m",
        "benchmark_hs300_return_6m",
        "benchmark_zz500_return",
        "benchmark_zz500_return_3m",
        "benchmark_zz500_return_6m",
        "excess_hs300_return",
        "excess_hs300_return_3m",
        "excess_hs300_return_6m",
        "excess_zz500_return",
        "excess_zz500_return_3m",
        "excess_zz500_return_6m",
        "turnover",
        "ic",
        "cumulative_net_value",
        "buy_codes",
        "sell_codes",
        "hold_codes",
    ]
    available_columns = [column for column in monthly_columns if column in backtest_df.columns]
    if "month" in backtest_df.columns and not backtest_df.empty:
        monthly_preview = backtest_df[available_columns].sort_values("month").copy()
    else:
        monthly_preview = backtest_df[available_columns].copy()
    diagnostics = _build_backtest_diagnostics(metrics, run_config, backtest_df, scores_path)

    monthly_row_html: list[str] = []
    if not monthly_preview.empty:
        for row in monthly_preview.itertuples(index=False):
            cells = []
            for column, value in zip(available_columns, row, strict=True):
                if column.endswith("_return"):
                    rendered = _format_pct(value)
                elif column in {"turnover", "ic", "cumulative_net_value"}:
                    rendered = _format_float(value)
                else:
                    rendered = str(value)
                cells.append(f"<td>{escape(rendered)}</td>")
            monthly_row_html.append("<tr>" + "".join(cells) + "</tr>")
    else:
        monthly_row_html.append(
            f"<tr><td colspan=\"{max(len(available_columns), 1)}\">暂无月度回测结果。</td></tr>"
        )

    monthly_headers = "".join(f"<th>{escape(name)}</th>" for name in available_columns)
    monthly_body = "\n".join(monthly_row_html)

    trade_signal_rows = _build_trade_signal_rows(backtest_df)
    trade_signal_row_html: list[str] = []
    for row in trade_signal_rows:
        trade_signal_row_html.append(
            "<tr>"
            f"<td>{escape(row['month'])}</td>"
            f"<td>{escape(row['buy_count'])}</td>"
            f"<td>{escape(row['buy_codes'])}</td>"
            f"<td>{escape(row['sell_count'])}</td>"
            f"<td>{escape(row['sell_codes'])}</td>"
            f"<td>{escape(row['hold_count'])}</td>"
            f"<td>{escape(row['hold_codes'])}</td>"
            "</tr>"
        )
    if not trade_signal_row_html:
        trade_signal_row_html.append(
            '<tr><td colspan="7">暂无买卖点数据。</td></tr>'
        )

    model_run_rows: list[str] = []
    for run in model_runs:
        model_run_rows.append(
            "<tr>"
            f"<td>{escape(str(run.get('run_id', '')))}</td>"
            f"<td>{escape(str(run.get('model_name', '')))}</td>"
            f"<td>{escape(str(run.get('epochs', '')))}</td>"
            f"<td>{escape(str(run.get('decay_months', '')))}</td>"
            f"<td>{escape(str(run.get('score_rows', '')))}</td>"
            f"<td>{escape(str(run.get('run_dir', '')))}</td>"
            "</tr>"
        )
    if not model_run_rows:
        model_run_rows.append(
            '<tr><td colspan="6">未在 model_runs 目录下找到模型运行元数据。</td></tr>'
        )

    net_value_data = json.dumps(_build_chart_series(backtest_df, "cumulative_net_value"), ensure_ascii=False)
    portfolio_return_data = json.dumps(_build_chart_series(backtest_df, "portfolio_return"), ensure_ascii=False)
    excess_hs300_data = json.dumps(_build_chart_series(backtest_df, "excess_hs300_return"), ensure_ascii=False)
    excess_zz500_data = json.dumps(_build_chart_series(backtest_df, "excess_zz500_return"), ensure_ascii=False)
    kline_trade_payload = json.dumps(
        _build_kline_trade_payload(backtest_df, resolved_daily_k_path),
        ensure_ascii=False,
    )
    gnn_graph_data = json.dumps(
        _build_gnn_graph_payload(
            scores_path=scores_path,
            stock_graph_edges_path=resolved_stock_graph_edges_path,
        ),
        ensure_ascii=False,
    )
    kg_graph_data = json.dumps(
        _build_kg_graph_payload(resolved_kg_stock_edges_llm_path),
        ensure_ascii=False,
    )
    monthly_rows_json = json.dumps(monthly_preview.to_dict(orient="records"), ensure_ascii=False, default=str)
    metric_rows_json = json.dumps(
        [{"name": name, "value": value} for name, value in metric_rows],
        ensure_ascii=False,
    )
    config_rows_json = json.dumps(
        [{"name": name, "value": _format_config_value(value)} for name, value in config_rows],
        ensure_ascii=False,
    )
    trade_signal_rows_json = json.dumps(trade_signal_rows, ensure_ascii=False)
    model_run_rows_json = json.dumps(model_runs, ensure_ascii=False, default=str)
    input_output_rows_json = json.dumps(
        [
            {"name": "scores_path", "value": str(scores_path)},
            {"name": "labels_path", "value": str(labels_path)},
            {"name": "benchmark_path", "value": str(benchmark_path)},
            {"name": "output_returns_path", "value": str(output_returns_path)},
            {"name": "output_metrics_path", "value": str(output_metrics_path)},
            {"name": "output_html_path", "value": str(output_path)},
            {"name": "stock_graph_edges_path", "value": str(resolved_stock_graph_edges_path)},
            {"name": "kg_stock_edges_llm_path", "value": str(resolved_kg_stock_edges_llm_path)},
        ],
        ensure_ascii=False,
    )
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_name_json = json.dumps(
        str(effective_run_config.get("experiment_name") or effective_run_config.get("config_name") or ""),
        ensure_ascii=False,
    )
    experiment_description_json = json.dumps(
        str(effective_run_config.get("experiment_description") or ""),
        ensure_ascii=False,
    )
    html = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>超图选股回测报告</title>
    <link rel=\"stylesheet\" href=\"https://unpkg.com/antd@5/dist/reset.css\" />
    <style>
        :root {{
            --page-bg: #f5f7fb;
            --panel-border: #e6ebf0;
            --text-muted: #667085;
        }}
        * {{ box-sizing: border-box; }}
        html, body {{
            width: 100%;
            max-width: 100%;
            overflow-x: hidden;
        }}
        body {{
            margin: 0;
            background: var(--page-bg);
            color: #172033;
            font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", \"PingFang SC\", \"Noto Sans SC\", sans-serif;
        }}
        #root {{ min-height: 100vh; }}
        .report-shell {{
            width: 100%;
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px clamp(8px, 2vw, 16px) 48px;
            min-width: 0;
        }}
        .report-stack {{ display: grid; gap: 16px; min-width: 0; }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            min-width: 0;
        }}
        .chart {{
            width: 100%;
            height: 320px;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
        }}
        .chart-tile {{ min-width: 0; }}
        .kline-toolbar, .graph-toolbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }}
        .tv-warning {{ display: none; margin-bottom: 10px; }}
        .gnn-graph {{
            width: 100%;
            height: 540px;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: #fff;
            overflow: hidden;
            touch-action: none;
        }}
        .gnn-graph svg {{ display: block; cursor: grab; }}
        .gnn-graph svg:active {{ cursor: grabbing; }}
        .kg-browser-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(360px, 0.75fr);
            gap: 12px;
            align-items: stretch;
        }}
        .kg-graph {{
            width: 100%;
            height: 620px;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: #fff;
            overflow: hidden;
            touch-action: none;
        }}
        .kg-graph svg {{ display: block; cursor: grab; }}
        .kg-graph svg:active {{ cursor: grabbing; }}
        .kg-side-panel {{
            min-width: 0;
            display: grid;
            gap: 12px;
            align-content: start;
        }}
        .kg-info-panel {{
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: #fff;
            padding: 12px;
            min-width: 0;
        }}
        .kg-info-title {{
            margin: 0 0 8px;
            font-size: 14px;
            font-weight: 600;
            color: #172033;
        }}
        .kg-type-list {{
            max-height: 210px;
            overflow: auto;
        }}
        .muted-note {{ color: var(--text-muted); font-size: 13px; }}
        .mono-cell {{
            overflow-wrap: anywhere;
            word-break: break-word;
        }}
        .ant-card, .ant-card-body {{
            border-radius: 8px;
            max-width: 100%;
            min-width: 0;
        }}
        .ant-tabs, .ant-tabs-content-holder, .ant-tabs-content, .ant-tabs-tabpane {{
            max-width: 100%;
            min-width: 0;
        }}
        .ant-table-wrapper, .ant-table, .ant-table-container {{
            max-width: 100%;
            min-width: 0;
        }}
        .ant-table-cell, .ant-descriptions-item-content {{
            overflow-wrap: anywhere;
            word-break: break-word;
        }}
        .ant-descriptions, .ant-descriptions-view, .ant-descriptions table {{
            max-width: 100%;
            table-layout: fixed;
        }}
        .graph-toolbar .ant-space, .kline-toolbar .ant-space {{
            max-width: 100%;
        }}
        .graph-toolbar .ant-select, .graph-toolbar .ant-input-affix-wrapper, .kline-toolbar .ant-select {{
            max-width: 100%;
        }}
        @media (max-width: 900px) {{
            .chart-grid {{ grid-template-columns: 1fr; }}
            .chart {{ height: 280px; }}
            .gnn-graph {{ height: 440px; }}
            .kg-browser-grid {{ grid-template-columns: 1fr; }}
            .kg-graph {{ height: 460px; }}
        }}
        @media (max-width: 640px) {{
            .report-shell {{ padding: 12px 8px 32px; }}
            .graph-toolbar, .kline-toolbar {{ align-items: stretch; }}
            .graph-toolbar > *, .kline-toolbar > * {{ width: 100%; }}
        }}
    </style>
</head>
<body>
    <div id=\"root\"></div>
    <script src=\"https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js\"></script>
    <script src=\"https://cdn.jsdelivr.net/npm/d3@7\"></script>
    <script crossorigin src=\"https://unpkg.com/react@18/umd/react.production.min.js\"></script>
    <script crossorigin src=\"https://unpkg.com/react-dom@18/umd/react-dom.production.min.js\"></script>
    <script src=\"https://unpkg.com/dayjs@1/dayjs.min.js\"></script>
    <script crossorigin src=\"https://unpkg.com/antd@5/dist/antd.min.js\"></script>
    <script>
        const netValueData = {net_value_data};
        const portfolioReturnData = {portfolio_return_data};
        const excessHs300Data = {excess_hs300_data};
        const excessZz500Data = {excess_zz500_data};
        const klineTradePayload = {kline_trade_payload};
        const gnnGraphData = {gnn_graph_data};
        const kgGraphData = {kg_graph_data};
        const metricRows = {metric_rows_json};
        const configRows = {config_rows_json};
        const monthlyRows = {monthly_rows_json};
        const tradeSignalRows = {trade_signal_rows_json};
        const modelRunRows = {model_run_rows_json};
        const inputOutputRows = {input_output_rows_json};
        const diagnostics = {json.dumps(diagnostics, ensure_ascii=False)};
        const generatedAt = {json.dumps(generated_at, ensure_ascii=False)};
        const experimentName = {experiment_name_json};
        const experimentDescription = {experiment_description_json};
        const modelRunsDir = {json.dumps(str(model_runs_dir), ensure_ascii=False)};
        const dailyKPath = {json.dumps(str(resolved_daily_k_path), ensure_ascii=False)};
        const {{ createElement: h }} = React;
        const {{ ConfigProvider, Alert, Button, Card, Col, Descriptions, Input, List, Row, Select, Space, Statistic, Table, Tabs, Tag, Typography }} = antd;
        const {{ Title, Paragraph, Text }} = Typography;

        const addLineSeriesCompat = (chart, chartLib, options) => {{
            if (typeof chart.addLineSeries === 'function') {{
                return chart.addLineSeries(options);
            }}
            return chart.addSeries(chartLib.LineSeries, options);
        }};

        const createLineChart = (containerId, title, data, color, valueFormatter) => {{
            const container = document.getElementById(containerId);
            if (!container || !data || data.length === 0) {{
                return null;
            }}
            const chartLib = window.LightweightCharts;
            const chart = chartLib.createChart(container, {{
                width: container.clientWidth,
                height: container.clientHeight,
                layout: {{ background: {{ color: '#ffffff' }}, textColor: '#334155' }},
                grid: {{ vertLines: {{ color: '#f1f5f9' }}, horzLines: {{ color: '#f1f5f9' }} }},
                crosshair: {{ mode: chartLib.CrosshairMode.Normal }},
                rightPriceScale: {{ borderColor: '#cbd5e1' }},
                timeScale: {{ borderColor: '#cbd5e1' }},
                localization: valueFormatter ? {{ priceFormatter: valueFormatter }} : undefined,
            }});
            const series = addLineSeriesCompat(chart, chartLib, {{ color, lineWidth: 2, title }});
            series.setData(data);
            chart.timeScale().fitContent();
            return {{ chart, container }};
        }};

        const addCandlestickSeriesCompat = (chart, chartLib, options) => {{
            if (chartLib.CandlestickSeries && typeof chart.addSeries === 'function') {{
                return chart.addSeries(chartLib.CandlestickSeries, options);
            }}
            return chart.addCandlestickSeries(options);
        }};

        const setSeriesMarkersCompat = (chartLib, series, markers) => {{
            if (typeof chartLib.createSeriesMarkers === 'function') {{
                return chartLib.createSeriesMarkers(series, markers);
            }}
            if (typeof series.setMarkers === 'function') {{
                series.setMarkers(markers);
            }}
            return null;
        }};

        let chartStates = [];
        let klineChartState = null;
        let gnnGraphState = null;
        let kgGraphState = null;

        const renderKlineTradeChart = (code) => {{
            const chartLib = window.LightweightCharts;
            const warning = document.getElementById('kline-warning');
            const summary = document.getElementById('kline-summary');
            const container = document.getElementById('chart-kline');
            if (!container || !summary) return;
            const seriesByCode = klineTradePayload.series_by_code || {{}};
            const codePayload = seriesByCode[code];
            container.innerHTML = '';
            klineChartState = null;
            if (!chartLib || typeof chartLib.createChart !== 'function' || !codePayload || !Array.isArray(codePayload.candles) || codePayload.candles.length === 0) {{
                if (warning) warning.style.display = 'block';
                summary.textContent = '暂无可展示的K线。';
                return;
            }}
            if (warning) warning.style.display = 'none';
            const chart = chartLib.createChart(container, {{
                width: container.clientWidth,
                height: container.clientHeight,
                layout: {{ background: {{ color: '#ffffff' }}, textColor: '#334155' }},
                grid: {{ vertLines: {{ color: '#f1f5f9' }}, horzLines: {{ color: '#f1f5f9' }} }},
                rightPriceScale: {{ borderColor: '#cbd5e1' }},
                timeScale: {{ borderColor: '#cbd5e1' }},
            }});
            const candleSeries = addCandlestickSeriesCompat(chart, chartLib, {{
                upColor: '#0f766e', downColor: '#be123c', borderVisible: false,
                wickUpColor: '#0f766e', wickDownColor: '#be123c',
            }});
            candleSeries.setData(codePayload.candles);
            setSeriesMarkersCompat(chartLib, candleSeries, codePayload.markers || []);
            chart.timeScale().fitContent();
            summary.textContent = `${{code}}：${{codePayload.candles.length}} 根K线，${{(codePayload.markers || []).length}} 个买卖标记。`;
            klineChartState = {{ chart, container }};
        }};

        const initKlineTradeChart = () => {{
            const select = document.getElementById('kline-code-select');
            const warning = document.getElementById('kline-warning');
            if (!select) return;
            const codes = Array.isArray(klineTradePayload.codes) ? klineTradePayload.codes : [];
            if (codes.length === 0) {{
                if (warning) warning.style.display = 'block';
                return;
            }}
            renderKlineTradeChart(select.value || codes[0]);
        }};

        const initCharts = () => {{
            if (!window.LightweightCharts || typeof window.LightweightCharts.createChart !== 'function') {{
                const warning = document.getElementById('tv-warning');
                if (warning) warning.style.display = 'block';
                return;
            }}
            const percentFormatter = (value) => `${{(value * 100).toFixed(2)}}%`;
            chartStates = [
                createLineChart('chart-net-value', '累计净值', netValueData, '#0f766e', null),
                createLineChart('chart-portfolio-return', '组合收益', portfolioReturnData, '#ea580c', percentFormatter),
                createLineChart('chart-excess-hs300', '超额收益HS300', excessHs300Data, '#1d4ed8', percentFormatter),
                createLineChart('chart-excess-zz500', '超额收益ZZ500', excessZz500Data, '#be123c', percentFormatter),
            ].filter(Boolean);
        }};

        const fitGnnGraph = (duration = 250) => {{
            if (!gnnGraphState) return;
            const {{ svg, viewport, zoom, nodes, width, height }} = gnnGraphState;
            if (!nodes.length) return;
            const xs = nodes.map((d) => Number(d.x)).filter(Number.isFinite);
            const ys = nodes.map((d) => Number(d.y)).filter(Number.isFinite);
            if (!xs.length || !ys.length) return;
            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);
            const graphWidth = Math.max(maxX - minX, 1);
            const graphHeight = Math.max(maxY - minY, 1);
            const scale = Math.max(0.2, Math.min(3, 0.86 / Math.max(graphWidth / width, graphHeight / height)));
            const tx = width / 2 - scale * (minX + maxX) / 2;
            const ty = height / 2 - scale * (minY + maxY) / 2;
            const transform = d3.zoomIdentity.translate(tx, ty).scale(scale);
            svg.transition().duration(duration).call(zoom.transform, transform);
        }};

        const zoomGnnGraph = (factor) => {{
            if (!gnnGraphState) return;
            gnnGraphState.svg.transition().duration(180).call(gnnGraphState.zoom.scaleBy, factor);
        }};

        const resetGnnGraph = () => {{
            if (!gnnGraphState) return;
            gnnGraphState.svg.transition().duration(220).call(gnnGraphState.zoom.transform, d3.zoomIdentity);
        }};

        const edgeTypeColor = (edgeType) => {{
            const palette = {{
                same_industry: '#0f766e',
                same_industry_topk: '#0f766e',
                market_corr: '#1d4ed8',
                market_corr_topk: '#1d4ed8',
                text_similarity: '#9333ea',
                kg_llm_core_subject: '#0891b2',
                kg_llm_peer: '#2563eb',
                kg_llm_positive_driver: '#16a34a',
                kg_llm_risk_pressure: '#ea580c',
            }};
            return palette[edgeType] || '#64748b';
        }};

        const renderGnnGraph = () => {{
            const warning = document.getElementById('gnn-warning');
            const summary = document.getElementById('gnn-graph-summary');
            const container = document.getElementById('gnn-graph');
            if (!container || !summary) return;
            if (!window.d3) {{
                if (warning) warning.style.display = 'block';
                summary.textContent = 'D3未加载，无法渲染GNN图。';
                return;
            }}
            const nodes = Array.isArray(gnnGraphData.nodes) ? gnnGraphData.nodes.map((d) => ({{ ...d }})) : [];
            const links = Array.isArray(gnnGraphData.links) ? gnnGraphData.links.map((d) => ({{ ...d }})) : [];
            summary.textContent = `最近月份: ${{gnnGraphData.month || '-'}}，节点数: ${{nodes.length}}，边数: ${{links.length}}。`;
            if (nodes.length === 0) {{
                container.innerHTML = '<div style="padding: 12px; color:#64748b;">没有可用于展示的GNN节点数据。</div>';
                gnnGraphState = null;
                return;
            }}
            container.innerHTML = '';
            const width = Math.max(container.clientWidth, 320);
            const height = Math.max(container.clientHeight, 320);
            const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
            const viewport = svg.append('g').attr('class', 'graph-viewport');
            const zoom = d3.zoom()
                .scaleExtent([0.2, 5])
                .on('zoom', (event) => viewport.attr('transform', event.transform));
            svg.call(zoom);

            const scoreAbsMax = d3.max(nodes, (d) => Math.abs(Number(d.score) || 0)) || 1;
            const radiusScale = d3.scaleLinear().domain([0, scoreAbsMax]).range([6, 16]);
            const colorScale = d3.scaleLinear().domain([-scoreAbsMax, 0, scoreAbsMax]).range(['#2563eb', '#94a3b8', '#dc2626']);
            const simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id((d) => d.id).distance(75).strength(0.35))
                .force('charge', d3.forceManyBody().strength(-180))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius((d) => radiusScale(Math.abs(Number(d.score) || 0)) + 4));
            const link = viewport.append('g')
                .attr('stroke-opacity', 0.45)
                .selectAll('line')
                .data(links)
                .join('line')
                .attr('stroke-width', (d) => 0.7 + Math.min(Math.abs(Number(d.edge_weight) || 0), 3.0))
                .attr('stroke', (d) => edgeTypeColor(d.edge_type));
            const node = viewport.append('g')
                .selectAll('circle')
                .data(nodes)
                .join('circle')
                .attr('r', (d) => radiusScale(Math.abs(Number(d.score) || 0)))
                .attr('fill', (d) => colorScale(Number(d.score) || 0))
                .attr('stroke', '#ffffff')
                .attr('stroke-width', 1.2)
                .call(d3.drag()
                    .on('start', (event, d) => {{
                        event.sourceEvent?.stopPropagation();
                        if (!event.active) simulation.alphaTarget(0.3).restart();
                        d.fx = d.x;
                        d.fy = d.y;
                    }})
                    .on('drag', (event, d) => {{
                        event.sourceEvent?.stopPropagation();
                        d.fx = event.x;
                        d.fy = event.y;
                    }})
                    .on('end', (event, d) => {{
                        event.sourceEvent?.stopPropagation();
                        if (!event.active) simulation.alphaTarget(0);
                        d.fx = null;
                        d.fy = null;
                    }}));
            node.append('title').text((d) => `code=${{d.id}}\nscore=${{(Number(d.score) || 0).toFixed(6)}}\nrank=${{d.rank}}`);
            const labels = viewport.append('g')
                .selectAll('text')
                .data(nodes)
                .join('text')
                .text((d) => d.id)
                .attr('font-size', 10)
                .attr('fill', '#334155')
                .attr('dx', 6)
                .attr('dy', 3)
                .style('pointer-events', 'none');
            simulation.on('tick', () => {{
                link.attr('x1', (d) => d.source.x).attr('y1', (d) => d.source.y).attr('x2', (d) => d.target.x).attr('y2', (d) => d.target.y);
                node.attr('cx', (d) => d.x).attr('cy', (d) => d.y);
                labels.attr('x', (d) => d.x).attr('y', (d) => d.y);
            }});
            gnnGraphState = {{ svg, viewport, zoom, simulation, nodes, width, height }};
            setTimeout(() => fitGnnGraph(0), 500);
        }};

        const filterKgEdges = (month, edgeType, query) => {{
            const rows = Array.isArray(kgGraphData.edge_rows) ? kgGraphData.edge_rows : [];
            const normalizedQuery = String(query || '').trim().toLowerCase();
            return rows.filter((row) => {{
                if (month !== 'all' && String(row.month || '') !== month) return false;
                if (edgeType !== 'all' && String(row.edge_type || '') !== edgeType) return false;
                if (!normalizedQuery) return true;
                return [row.src_code, row.dst_code, row.source_id, row.report_id, row.edge_type]
                    .some((value) => String(value || '').toLowerCase().includes(normalizedQuery));
            }});
        }};

        const summarizeKgEdges = (edges) => {{
            const nodes = new Set();
            const typeCounts = {{}};
            edges.forEach((edge) => {{
                if (edge.src_code) nodes.add(edge.src_code);
                if (edge.dst_code) nodes.add(edge.dst_code);
                const edgeType = String(edge.edge_type || 'unknown');
                typeCounts[edgeType] = (typeCounts[edgeType] || 0) + 1;
            }});
            return {{ nodeCount: nodes.size, edgeCount: edges.length, typeCounts }};
        }};

        const fitKgGraph = (duration = 250) => {{
            if (!kgGraphState) return;
            const {{ svg, zoom, nodes, width, height }} = kgGraphState;
            if (!nodes.length) return;
            const xs = nodes.map((d) => Number(d.x)).filter(Number.isFinite);
            const ys = nodes.map((d) => Number(d.y)).filter(Number.isFinite);
            if (!xs.length || !ys.length) return;
            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);
            const graphWidth = Math.max(maxX - minX, 1);
            const graphHeight = Math.max(maxY - minY, 1);
            const scale = Math.max(0.18, Math.min(3, 0.86 / Math.max(graphWidth / width, graphHeight / height)));
            const tx = width / 2 - scale * (minX + maxX) / 2;
            const ty = height / 2 - scale * (minY + maxY) / 2;
            svg.transition().duration(duration).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
        }};

        const zoomKgGraph = (factor) => {{
            if (!kgGraphState) return;
            kgGraphState.svg.transition().duration(180).call(kgGraphState.zoom.scaleBy, factor);
        }};

        const resetKgGraph = () => {{
            if (!kgGraphState) return;
            kgGraphState.svg.transition().duration(220).call(kgGraphState.zoom.transform, d3.zoomIdentity);
        }};

        const renderKgGraph = (edges) => {{
            const warning = document.getElementById('kg-warning');
            const summary = document.getElementById('kg-graph-summary');
            const container = document.getElementById('kg-graph');
            if (!container || !summary) return;
            if (!window.d3) {{
                if (warning) warning.style.display = 'block';
                summary.textContent = 'D3未加载，无法渲染KG图。';
                return;
            }}
            if (warning) warning.style.display = 'none';
            const maxGraphEdges = 500;
            const graphEdges = (Array.isArray(edges) ? edges : []).slice(0, maxGraphEdges);
            const stats = summarizeKgEdges(edges || []);
            summary.textContent = `筛选后节点: ${{stats.nodeCount}}，边: ${{stats.edgeCount}}，当前绘图: ${{graphEdges.length}} 条边。`;
            container.innerHTML = '';
            kgGraphState = null;
            if (graphEdges.length === 0) {{
                container.innerHTML = '<div style="padding: 12px; color:#64748b;">没有符合筛选条件的KG边。</div>';
                return;
            }}

            const degree = {{}};
            graphEdges.forEach((edge) => {{
                degree[edge.src_code] = (degree[edge.src_code] || 0) + 1;
                degree[edge.dst_code] = (degree[edge.dst_code] || 0) + 1;
            }});
            const nodes = Object.keys(degree).sort().map((code) => ({{ id: code, degree: degree[code] }}));
            const links = graphEdges.map((edge) => ({{
                source: edge.src_code,
                target: edge.dst_code,
                edge_type: edge.edge_type,
                edge_weight: Number(edge.edge_weight) || 0,
                report_id: edge.report_id,
                source_id: edge.source_id,
                month: edge.month,
            }}));
            const width = Math.max(container.clientWidth, 320);
            const height = Math.max(container.clientHeight, 360);
            const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
            const viewport = svg.append('g').attr('class', 'kg-graph-viewport');
            const zoom = d3.zoom()
                .scaleExtent([0.15, 5])
                .on('zoom', (event) => viewport.attr('transform', event.transform));
            svg.call(zoom);

            const maxDegree = d3.max(nodes, (d) => Number(d.degree) || 0) || 1;
            const radiusScale = d3.scaleSqrt().domain([1, maxDegree]).range([6, 18]);
            const simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id((d) => d.id).distance(88).strength(0.28))
                .force('charge', d3.forceManyBody().strength(-210))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius((d) => radiusScale(Number(d.degree) || 1) + 5));
            const link = viewport.append('g')
                .attr('stroke-opacity', 0.52)
                .selectAll('line')
                .data(links)
                .join('line')
                .attr('stroke-width', (d) => 0.8 + Math.min(Math.max(Number(d.edge_weight) || 0, 0), 2.4))
                .attr('stroke', (d) => edgeTypeColor(d.edge_type));
            link.append('title').text((d) => `${{d.source.id || d.source}} -> ${{d.target.id || d.target}}\n${{d.edge_type}}\nweight=${{(Number(d.edge_weight) || 0).toFixed(4)}}\nreport=${{d.report_id || '-'}}`);
            const node = viewport.append('g')
                .selectAll('circle')
                .data(nodes)
                .join('circle')
                .attr('r', (d) => radiusScale(Number(d.degree) || 1))
                .attr('fill', '#ffffff')
                .attr('stroke', '#1677ff')
                .attr('stroke-width', 1.8)
                .call(d3.drag()
                    .on('start', (event, d) => {{
                        event.sourceEvent?.stopPropagation();
                        if (!event.active) simulation.alphaTarget(0.3).restart();
                        d.fx = d.x;
                        d.fy = d.y;
                    }})
                    .on('drag', (event, d) => {{
                        event.sourceEvent?.stopPropagation();
                        d.fx = event.x;
                        d.fy = event.y;
                    }})
                    .on('end', (event, d) => {{
                        event.sourceEvent?.stopPropagation();
                        if (!event.active) simulation.alphaTarget(0);
                        d.fx = null;
                        d.fy = null;
                    }}));
            node.append('title').text((d) => `code=${{d.id}}\ndegree=${{d.degree}}`);
            const labels = viewport.append('g')
                .selectAll('text')
                .data(nodes)
                .join('text')
                .text((d) => d.id)
                .attr('font-size', 10)
                .attr('fill', '#172033')
                .attr('dx', 7)
                .attr('dy', 3)
                .style('pointer-events', 'none');
            simulation.on('tick', () => {{
                link.attr('x1', (d) => d.source.x).attr('y1', (d) => d.source.y).attr('x2', (d) => d.target.x).attr('y2', (d) => d.target.y);
                node.attr('cx', (d) => d.x).attr('cy', (d) => d.y);
                labels.attr('x', (d) => d.x).attr('y', (d) => d.y);
            }});
            kgGraphState = {{ svg, viewport, zoom, simulation, nodes, width, height }};
            setTimeout(() => fitKgGraph(0), 550);
        }};

        const renderAllVisuals = () => {{
            initCharts();
            initKlineTradeChart();
            renderGnnGraph();
        }};

        const formatMetricValue = (name, value) => name.includes('收益') ? value : value;
        const overviewMetricNames = [
            '组合总收益',
            '沪深300总收益',
            '中证500总收益',
            '相对沪深300超额',
            '相对中证500超额',
            '平均换手率',
            '平均IC',
            '夏普比率(年化)',
        ];
        const metricStats = metricRows.filter((row) => overviewMetricNames.includes(row.name));
        const hasMonthlyRows = Array.isArray(monthlyRows) && monthlyRows.length > 0;
        const monthlyColumns = Object.keys(monthlyRows[0] || {{}}).map((key) => ({{
            title: key,
            dataIndex: key,
            width: key.includes('codes') ? 260 : 130,
            render: (value) => String(value ?? ''),
        }}));
        const tradeColumns = [
            {{ title: '月份', dataIndex: 'month', width: 100 }},
            {{ title: '买入数量', dataIndex: 'buy_count', width: 100 }},
            {{ title: '买入代码', dataIndex: 'buy_codes', width: 280 }},
            {{ title: '卖出数量', dataIndex: 'sell_count', width: 100 }},
            {{ title: '卖出代码', dataIndex: 'sell_codes', width: 280 }},
            {{ title: '继续持有数量', dataIndex: 'hold_count', width: 120 }},
            {{ title: '继续持有代码', dataIndex: 'hold_codes', width: 280 }},
        ];
        const modelRunColumns = [
            {{ title: 'run_id', dataIndex: 'run_id', width: 180 }},
            {{ title: 'model_name', dataIndex: 'model_name', width: 260 }},
            {{ title: 'epochs', dataIndex: 'epochs', width: 90 }},
            {{ title: 'decay_months', dataIndex: 'decay_months', width: 130 }},
            {{ title: 'score_rows', dataIndex: 'score_rows', width: 110 }},
            {{ title: 'run_dir', dataIndex: 'run_dir', width: 360, render: (value) => h('span', {{ className: 'mono-cell' }}, String(value ?? '')) }},
        ];
        const kgEdgeColumns = [
            {{ title: '月份', dataIndex: 'month', width: 100 }},
            {{ title: '源股票', dataIndex: 'src_code', width: 120, render: (value) => h(Tag, {{ color: 'blue' }}, String(value ?? '')) }},
            {{ title: '目标股票', dataIndex: 'dst_code', width: 120, render: (value) => h(Tag, {{ color: 'geekblue' }}, String(value ?? '')) }},
            {{ title: '边类型', dataIndex: 'edge_type', width: 230, render: (value) => h(Tag, {{ color: 'default', style: {{ borderColor: edgeTypeColor(value), color: edgeTypeColor(value) }} }}, String(value ?? '')) }},
            {{ title: '权重', dataIndex: 'edge_weight', width: 100, render: (value) => (Number(value) || 0).toFixed(4) }},
            {{ title: 'report_id', dataIndex: 'report_id', width: 160, render: (value) => h('span', {{ className: 'mono-cell' }}, String(value ?? '')) }},
            {{ title: 'source_id', dataIndex: 'source_id', width: 280, render: (value) => h('span', {{ className: 'mono-cell' }}, String(value ?? '')) }},
        ];

        function App() {{
            const codes = Array.isArray(klineTradePayload.codes) ? klineTradePayload.codes : [];
            const [klineCode, setKlineCode] = React.useState(codes[0] || '');
            const [kgMonth, setKgMonth] = React.useState('all');
            const [kgEdgeType, setKgEdgeType] = React.useState('all');
            const [kgQuery, setKgQuery] = React.useState('');
            const kgFilteredEdges = React.useMemo(
                () => filterKgEdges(kgMonth, kgEdgeType, kgQuery),
                [kgMonth, kgEdgeType, kgQuery],
            );
            const kgFilteredSummary = React.useMemo(() => summarizeKgEdges(kgFilteredEdges), [kgFilteredEdges]);
            React.useEffect(() => {{
                renderAllVisuals();
                const handleResize = () => {{
                    chartStates.forEach((item) => {{
                        item.chart.applyOptions({{ width: item.container.clientWidth }});
                        item.chart.timeScale().fitContent();
                    }});
                    if (klineChartState) {{
                        klineChartState.chart.applyOptions({{ width: klineChartState.container.clientWidth }});
                        klineChartState.chart.timeScale().fitContent();
                    }}
                    renderGnnGraph();
                    renderKgGraph(kgFilteredEdges);
                }};
                window.addEventListener('resize', handleResize);
                return () => window.removeEventListener('resize', handleResize);
            }}, []);
            React.useEffect(() => {{
                if (klineCode) renderKlineTradeChart(klineCode);
            }}, [klineCode]);
            React.useEffect(() => {{
                renderKgGraph(kgFilteredEdges);
            }}, [kgFilteredEdges]);
            return h(ConfigProvider, {{ theme: {{ token: {{ colorPrimary: '#1677ff', borderRadius: 8 }} }} }},
                h('main', {{ className: 'report-shell' }},
                    h('div', {{ className: 'report-stack' }},
                        h(Card, null,
                            h(Space, {{ direction: 'vertical', size: 8 }},
                                h(Title, {{ level: 2, style: {{ margin: 0 }} }}, '超图选股回测报告'),
                                h(Space, {{ wrap: true }}, experimentName && h(Tag, {{ color: 'green' }}, `实验项目: ${{experimentName}}`), h(Tag, {{ color: 'blue' }}, `生成时间: ${{generatedAt}}`), h(Tag, null, 'TradingView Lightweight Charts'), h(Tag, null, 'D3 Force Graph')),
                                h(Paragraph, {{ className: 'muted-note', style: {{ margin: 0 }} }}, experimentDescription || '本报告包含回测指标、月度明细和模型文件说明。'),
                            ),
                        ),
                        h(Card, {{ title: '结果总览' }},
                            h(Row, {{ gutter: [12, 12] }}, metricStats.map((row) =>
                                h(Col, {{ xs: 12, md: 8, lg: 4, key: row.name }},
                                    h(Statistic, {{ title: row.name, value: formatMetricValue(row.name, row.value), valueStyle: {{ fontSize: 20 }} }}),
                                ),
                            )),
                        ),
                        diagnostics.length > 0 && h(Card, {{ title: '诊断信息' }},
                            h(Alert, {{ type: 'info', showIcon: true, message: diagnostics[0].title, description: diagnostics[0].description }}),
                            h(List, {{ size: 'small', dataSource: diagnostics.slice(1), renderItem: (item) => h(List.Item, null, h(Space, {{ direction: 'vertical', size: 0 }}, h(Text, {{ strong: true }}, item.title), h(Text, {{ className: 'muted-note' }}, item.description))) }}),
                        ),
                        h(Card, {{ title: '运行配置' }},
                            h(Descriptions, {{ size: 'small', column: {{ xs: 1, md: 2, xl: 3 }}, bordered: true, items: configRows.map((row) => ({{ key: row.name, label: row.name, children: String(row.value) }})) }}),
                        ),
                        h(Card, {{ title: '可视化曲线' }},
                            h(Alert, {{ id: 'tv-warning', className: 'tv-warning', type: 'warning', showIcon: true, title: '未加载 Lightweight Charts，无法显示曲线。请检查网络或 CDN 访问权限。' }}),
                            h('div', {{ className: 'chart-grid' }},
                                h(Card, {{ size: 'small', title: '组合累计净值', className: 'chart-tile' }}, h('div', {{ id: 'chart-net-value', className: 'chart' }})),
                                h(Card, {{ size: 'small', title: '月度组合收益', className: 'chart-tile' }}, h('div', {{ id: 'chart-portfolio-return', className: 'chart' }})),
                                h(Card, {{ size: 'small', title: '月度超额收益(相对沪深300)', className: 'chart-tile' }}, h('div', {{ id: 'chart-excess-hs300', className: 'chart' }})),
                                h(Card, {{ size: 'small', title: '月度超额收益(相对中证500)', className: 'chart-tile' }}, h('div', {{ id: 'chart-excess-zz500', className: 'chart' }})),
                            ),
                        ),
                        h(Card, {{ title: '月度回测明细' }},
                            hasMonthlyRows
                                ? h(Table, {{ rowKey: (_, index) => `month-${{index}}`, size: 'small', columns: monthlyColumns, dataSource: monthlyRows, scroll: {{ x: 'max-content' }}, pagination: {{ pageSize: 12 }} }})
                                : h(Alert, {{ type: 'info', showIcon: true, message: '暂无月度回测结果', description: '当前输入月份不足，已切换为诊断报告。' }})
                        ),
                        h(Card, {{ title: '买卖点表格' }}, h(Table, {{ rowKey: 'month', size: 'small', columns: tradeColumns, dataSource: tradeSignalRows, scroll: {{ x: 'max-content' }}, pagination: false }})),
                        h(Card, {{ title: 'K线买卖点' }},
                            h(Alert, {{ id: 'kline-warning', className: 'tv-warning', type: 'warning', showIcon: true, title: '没有可展示的K线买卖点数据，或 Lightweight Charts 未加载。' }}),
                            h('div', {{ className: 'kline-toolbar' }},
                                h(Space, {{ align: 'center' }},
                                    h(Text, null, '股票代码'),
                                    h(Select, {{ id: 'kline-code-select', value: klineCode, style: {{ width: 180 }}, onChange: setKlineCode, options: codes.map((code) => ({{ value: code, label: code }})) }}),
                                ),
                                h(Text, {{ id: 'kline-summary', className: 'muted-note' }}),
                            ),
                            h('div', {{ id: 'chart-kline', className: 'chart' }}),
                            h(Paragraph, {{ className: 'muted-note', style: {{ marginTop: 8, marginBottom: 0 }} }}, `K线数据路径: ${{dailyKPath}}`),
                        ),
                        h(Card, {{ title: 'GNN结构可视化' }},
                            h(Alert, {{ id: 'gnn-warning', className: 'tv-warning', type: 'warning', showIcon: true, title: '未加载 D3，无法显示GNN关系图。请检查网络或 CDN 访问权限。' }}),
                            h('div', {{ className: 'graph-toolbar' }},
                                h(Text, {{ id: 'gnn-graph-summary', className: 'muted-note' }}),
                                h(Space, {{ wrap: true }},
                                    h(Button, {{ size: 'small', onClick: () => zoomGnnGraph(1.25) }}, '放大'),
                                    h(Button, {{ size: 'small', onClick: () => zoomGnnGraph(0.8) }}, '缩小'),
                                    h(Button, {{ size: 'small', onClick: resetGnnGraph }}, '重置'),
                                    h(Button, {{ size: 'small', type: 'primary', onClick: () => fitGnnGraph(220) }}, '适配视图'),
                                ),
                            ),
                            h('div', {{ id: 'gnn-graph', className: 'gnn-graph' }}),
                        ),
                        h(Card, {{ title: '完整KG浏览器' }},
                            h(Alert, {{ id: 'kg-warning', className: 'tv-warning', type: 'warning', showIcon: true, title: '未加载 D3，无法显示KG图。请检查网络或 CDN 访问权限。' }}),
                            h(Row, {{ gutter: [12, 12], style: {{ marginBottom: 12 }} }},
                                h(Col, {{ xs: 12, md: 6 }},
                                    h(Statistic, {{ title: 'KG节点', value: kgGraphData.summary?.node_count || 0, valueStyle: {{ fontSize: 20 }} }}),
                                ),
                                h(Col, {{ xs: 12, md: 6 }},
                                    h(Statistic, {{ title: 'KG边', value: kgGraphData.summary?.edge_count || 0, valueStyle: {{ fontSize: 20 }} }}),
                                ),
                                h(Col, {{ xs: 12, md: 6 }},
                                    h(Statistic, {{ title: '筛选节点', value: kgFilteredSummary.nodeCount, valueStyle: {{ fontSize: 20 }} }}),
                                ),
                                h(Col, {{ xs: 12, md: 6 }},
                                    h(Statistic, {{ title: '筛选边', value: kgFilteredSummary.edgeCount, valueStyle: {{ fontSize: 20 }} }}),
                                ),
                            ),
                            h('div', {{ className: 'graph-toolbar' }},
                                h(Space, {{ wrap: true, align: 'center' }},
                                    h(Select, {{
                                        id: 'kg-month-select',
                                        value: kgMonth,
                                        style: {{ width: 160 }},
                                        onChange: setKgMonth,
                                        options: [{{ value: 'all', label: '全部月份' }}, ...(kgGraphData.months || []).map((month) => ({{ value: month, label: month }}))],
                                    }}),
                                    h(Select, {{
                                        id: 'kg-edge-type-select',
                                        value: kgEdgeType,
                                        style: {{ width: 260 }},
                                        onChange: setKgEdgeType,
                                        options: [{{ value: 'all', label: '全部边类型' }}, ...(kgGraphData.edge_types || []).map((edgeType) => ({{ value: edgeType, label: edgeType }}))],
                                    }}),
                                    h(Input, {{
                                        id: 'kg-search-input',
                                        allowClear: true,
                                        placeholder: '搜索股票/报告/source_id',
                                        value: kgQuery,
                                        onChange: (event) => setKgQuery(event.target.value),
                                        style: {{ width: 260 }},
                                    }}),
                                ),
                                h(Space, {{ wrap: true }},
                                    h(Button, {{ size: 'small', onClick: () => zoomKgGraph(1.25) }}, '放大'),
                                    h(Button, {{ size: 'small', onClick: () => zoomKgGraph(0.8) }}, '缩小'),
                                    h(Button, {{ size: 'small', onClick: resetKgGraph }}, '重置'),
                                    h(Button, {{ size: 'small', type: 'primary', onClick: () => fitKgGraph(220) }}, '适配视图'),
                                ),
                            ),
                            h('div', {{ className: 'kg-browser-grid' }},
                                h('div', null,
                                    h(Text, {{ id: 'kg-graph-summary', className: 'muted-note' }}),
                                    h('div', {{ id: 'kg-graph', className: 'kg-graph', style: {{ marginTop: 8 }} }}),
                                    h(Paragraph, {{ className: 'muted-note', style: {{ marginTop: 8, marginBottom: 0 }} }}, '图视图最多绘制当前筛选结果前500条边；下方表格保留完整筛选结果。'),
                                ),
                                h('div', {{ className: 'kg-side-panel' }},
                                    h('section', {{ className: 'kg-info-panel' }},
                                        h('h3', {{ className: 'kg-info-title' }}, '边类型分布'),
                                        h('div', {{ className: 'kg-type-list' }},
                                            Object.entries(kgFilteredSummary.typeCounts).length
                                                ? Object.entries(kgFilteredSummary.typeCounts).sort((a, b) => b[1] - a[1]).map(([edgeType, count]) =>
                                                    h('div', {{ key: edgeType, style: {{ display: 'flex', justifyContent: 'space-between', gap: 8, marginBottom: 6 }} }},
                                                        h(Tag, {{ style: {{ borderColor: edgeTypeColor(edgeType), color: edgeTypeColor(edgeType), marginInlineEnd: 0 }} }}, edgeType),
                                                        h(Text, null, count),
                                                    ),
                                                )
                                                : h(Text, {{ className: 'muted-note' }}, '暂无边类型。'),
                                        ),
                                    ),
                                    h('section', {{ className: 'kg-info-panel' }},
                                        h('h3', {{ className: 'kg-info-title' }}, 'KG文件'),
                                        h(Text, {{ className: 'mono-cell' }}, kgGraphData.kg_stock_edges_llm_path || ''),
                                    ),
                                ),
                            ),
                            h('div', {{ id: 'kg-edge-table', style: {{ marginTop: 12 }} }},
                                h(Table, {{
                                    rowKey: (row, index) => row.id || `kg-row-${{index}}`,
                                    size: 'small',
                                    columns: kgEdgeColumns,
                                    dataSource: kgFilteredEdges,
                                    scroll: {{ x: 'max-content', y: 420 }},
                                    pagination: {{ pageSize: 20, showSizeChanger: true }},
                                }}),
                            ),
                        ),
                        h(Card, {{ title: '模型文件说明(model_runs)' }},
                            h(Paragraph, {{ className: 'muted-note' }}, `模型运行目录: ${{modelRunsDir}}`),
                            h(Table, {{ rowKey: (row, index) => row.run_id || `run-${{index}}`, size: 'small', columns: modelRunColumns, dataSource: modelRunRows, scroll: {{ x: 'max-content' }}, pagination: false }}),
                        ),
                        h(Card, {{ title: '输入与输出文件' }},
                            h(Descriptions, {{ size: 'small', column: 1, bordered: true, items: inputOutputRows.map((row) => ({{ key: row.name, label: row.name, children: h('span', {{ className: 'mono-cell' }}, row.value) }})) }}),
                        ),
                    ),
                ),
            );
        }}

        ReactDOM.createRoot(document.getElementById('root')).render(h(App));
    </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _read_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_pickle(path)


if __name__ == "__main__":
    main()
