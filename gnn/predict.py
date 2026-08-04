from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from alphagraph.model.core import (
    FEATURE_SCHEMA_VERSION,
    NODE_FEATURE_COLUMNS,
    DualChannelSelector,
    build_monthly_samples,
    score_month,
    train_final_model,
)
from alphagraph.trading.strategies import apply_stock_filters, build_topk_portfolio


def _normalize_month(value: str) -> str:
    try:
        return str(pd.Period(str(value), freq="M"))
    except Exception as exc:
        raise ValueError(f"Invalid month: {value}") from exc


def select_training_months(
    monthly_samples: dict[str, dict[str, Any]],
    labels_df: pd.DataFrame,
    signal_month: str,
    train_start_month: str,
    train_end_month: str,
    training_label: str,
) -> list[str]:
    signal = _normalize_month(signal_month)
    start = _normalize_month(train_start_month)
    end = _normalize_month(train_end_month)
    if start > end:
        raise ValueError("train_start_month must not be after train_end_month")
    if end >= signal:
        raise ValueError("train_end_month must be before signal_month")
    if training_label not in labels_df.columns:
        raise ValueError(f"Training label column not found: {training_label}")

    requested_months = [
        str(month)
        for month in pd.period_range(
            pd.Period(start, freq="M"),
            pd.Period(end, freq="M"),
            freq="M",
        )
    ]
    normalized_samples = {
        _normalize_month(month): sample for month, sample in monthly_samples.items()
    }
    missing_samples = [month for month in requested_months if month not in normalized_samples]
    if missing_samples:
        raise ValueError(f"Training range missing samples: {missing_samples}")

    normalized_labels = labels_df.loc[labels_df[training_label].notna()].copy()
    normalized_labels["month"] = (
        normalized_labels["month"].astype(str).map(_normalize_month)
    )
    missing_labels: list[str] = []
    for month in requested_months:
        sample_codes = set(map(str, normalized_samples[month]["stock_codes"]))
        month_label_codes = set(
            normalized_labels.loc[normalized_labels["month"] == month, "code"].astype(str)
        )
        if not sample_codes.intersection(month_label_codes):
            missing_labels.append(month)
    if missing_labels:
        raise ValueError(
            f"Training range missing labels for {training_label}: {missing_labels}"
        )
    return requested_months


def checkpoint_path_for_training_end(
    checkpoint_dir: Path,
    train_end_month: str,
    as_of_date: str,
) -> Path:
    period = pd.Period(_normalize_month(train_end_month), freq="M")
    quarter = (period.month - 1) // 3 + 1
    resolved_as_of = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(resolved_as_of):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    checkpoint_available_as_of = (period + 1).end_time.normalize()
    if resolved_as_of.normalize() < checkpoint_available_as_of:
        raise ValueError(
            "as_of_date is earlier than the one-month label availability date: "
            f"{checkpoint_available_as_of.date().isoformat()}"
        )
    return Path(checkpoint_dir) / (
        f"checkpoint_{period.year}Q{quarter}_asof_{checkpoint_available_as_of:%Y%m%d}.pt"
    )


def build_checkpoint_metadata(
    model: DualChannelSelector,
    train_months: list[str],
    train_start_month: str,
    train_end_month: str,
    as_of_date: str,
    signal_month: str,
    seed: int,
    fusion_type: str,
    graph_module_type: str,
    edge_type_prior_mode: str,
    training_objective: str,
    training_label: str,
    edge_type_mapping: dict[str, int],
    industry_mapping: dict[str, int],
    alpha_factor_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stock_graph_module = model.stock_graph_module
    metadata = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "node_feature_columns": list(NODE_FEATURE_COLUMNS),
        "node_feature_dim": int(model.node_encoder.in_features),
        "hyperedge_feature_dim": int(model.hyperedge_encoder.in_features),
        "hidden_dim": int(model.node_encoder.out_features),
        "num_industries": int(model.industry_embedding.num_embeddings),
        "num_edge_types": int(getattr(stock_graph_module, "num_edge_types", 1)),
        "fusion_type": str(fusion_type),
        "graph_module_type": str(graph_module_type),
        "edge_type_prior_mode": str(edge_type_prior_mode),
        "training_objective": str(training_objective),
        "training_label": str(training_label),
        "train_months": [_normalize_month(month) for month in train_months],
        "train_start_month": _normalize_month(train_start_month),
        "train_end_month": _normalize_month(train_end_month),
        "signal_month": _normalize_month(signal_month),
        "as_of_date": pd.Timestamp(as_of_date).date().isoformat(),
        "seed": int(seed),
        "edge_type_mapping": dict(edge_type_mapping),
        "industry_mapping": dict(industry_mapping),
    }
    if alpha_factor_manifest is not None:
        metadata["alpha_factor_manifest"] = dict(alpha_factor_manifest)
    return metadata


def save_prediction_checkpoint(
    model: DualChannelSelector,
    metadata: dict[str, Any],
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": metadata},
        output_path,
    )


def load_prediction_checkpoint(
    checkpoint_path: Path,
    expected: dict[str, Any] | None = None,
) -> tuple[DualChannelSelector, dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    for key, expected_value in (expected or {}).items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"Checkpoint {key} mismatch: expected {expected_value!r}, "
                f"got {metadata.get(key)!r}"
            )

    model = DualChannelSelector(
        node_feature_dim=int(metadata["node_feature_dim"]),
        hyperedge_feature_dim=int(metadata["hyperedge_feature_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        num_industries=int(metadata["num_industries"]),
        fusion_type=str(metadata["fusion_type"]),
        graph_module_type=str(metadata["graph_module_type"]),
        num_edge_types=int(metadata["num_edge_types"]),
        edge_type_prior_mode=str(metadata["edge_type_prior_mode"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, metadata


def resolve_checkpoint_output_path(
    canonical_path: Path,
    force_retrain: bool,
    run_tag: str,
) -> Path:
    canonical_path = Path(canonical_path)
    if force_retrain and canonical_path.exists():
        return canonical_path.with_name(
            f"{canonical_path.stem}_rerun_{run_tag}{canonical_path.suffix}"
        )
    return canonical_path


def build_recommendations(
    scores_df: pd.DataFrame,
    monthly_features_df: pd.DataFrame,
    stock_basic_df: pd.DataFrame | None,
    topk: int,
    liquidity_quantile: float,
    min_price: float,
    exclude_st: bool,
    as_of_date: str,
) -> pd.DataFrame:
    filtered = apply_stock_filters(
        scores_df,
        monthly_features_df,
        liquidity_quantile=liquidity_quantile,
        min_price=min_price,
        exclude_st=exclude_st,
    )
    recommendations = build_topk_portfolio(filtered, k=topk)
    detail_columns = [
        column
        for column in ("industry", "ret_20d", "ret_60d")
        if column in monthly_features_df.columns and column not in recommendations.columns
    ]
    if detail_columns:
        details = monthly_features_df[["month", "code", *detail_columns]].drop_duplicates(
            subset=["month", "code"]
        )
        recommendations = recommendations.merge(details, on=["month", "code"], how="left")
    if stock_basic_df is not None and not stock_basic_df.empty and "code" in stock_basic_df.columns:
        name_column = next(
            (column for column in ("code_name", "name", "stock_name") if column in stock_basic_df.columns),
            None,
        )
        if name_column:
            names = stock_basic_df[["code", name_column]].drop_duplicates(subset=["code"])
            recommendations = recommendations.merge(names, on="code", how="left")
            recommendations = recommendations.rename(columns={name_column: "name"})
    if "name" not in recommendations.columns:
        recommendations["name"] = ""

    resolved_as_of = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(resolved_as_of):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    execution_from = (resolved_as_of + pd.offsets.BDay(1)).date().isoformat()
    recommendations["signal_month"] = recommendations["month"].astype(str)
    recommendations["as_of_date"] = resolved_as_of.date().isoformat()
    recommendations["execution_from"] = execution_from

    preferred_columns = [
        "rank",
        "code",
        "name",
        "score",
        "signal_month",
        "as_of_date",
        "execution_from",
        "industry",
        "month_end_close",
        "amount_20d_mean",
        "ret_20d",
        "ret_60d",
        "is_tradable",
        "is_st",
    ]
    ordered = [column for column in preferred_columns if column in recommendations.columns]
    trailing = [column for column in recommendations.columns if column not in ordered and column != "month"]
    return recommendations[ordered + trailing].reset_index(drop=True)


def run_prediction(
    artifact_dir: Path,
    signal_month: str,
    as_of_date: str,
    train_start_month: str,
    train_end_month: str,
    checkpoint_cadence_months: int = 3,
    topk: int = 20,
    force_retrain: bool = False,
    epochs: int = 20,
    decay_months: int = 3,
    fusion_type: str = "cross_encoder",
    graph_module_type: str = "relation_aware",
    edge_type_prior_mode: str = "none",
    community_reg_weight: float = 0.0,
    training_objective: str = "ranknet_pairwise",
    training_label: str = "future_excess_return_1m",
    liquidity_quantile: float = 0.2,
    min_price: float = 2.0,
    exclude_st: bool = True,
    seed: int = 42,
    model_name: str = "",
    run_tag: str | None = None,
) -> dict[str, Path]:
    if int(checkpoint_cadence_months) != 3:
        raise ValueError("checkpoint_cadence_months must be 3 for quarterly checkpoints")
    artifact_dir = Path(artifact_dir)
    signal = _normalize_month(signal_month)
    run_tag = run_tag or datetime.now().strftime("%Y_%m_%d-%H%M%S")

    reports_df = _read_required_frame(artifact_dir / "reports.parquet")
    incidence_df = _read_required_frame(artifact_dir / "incidence.parquet")
    monthly_features_df = _read_required_frame(artifact_dir / "monthly_stock_features.parquet")
    labels_df = _read_required_frame(artifact_dir / "monthly_labels.parquet")
    embeddings_df = _read_required_frame(artifact_dir / "report_embeddings.parquet")
    stock_edges_df = _read_required_frame(artifact_dir / "stock_graph_edges.parquet")
    stock_basic_df = _read_optional_frame(artifact_dir / "stock_basic.parquet")
    stock_clusters_df = _read_optional_frame(artifact_dir / "stock_clusters.parquet")
    cluster_edges_df = _read_optional_frame(artifact_dir / "cluster_graph_edges.parquet")
    alpha_manifest_path = artifact_dir / "alpha_factor_manifest.json"
    alpha_factor_manifest = (
        json.loads(alpha_manifest_path.read_text(encoding="utf-8"))
        if alpha_manifest_path.exists()
        else None
    )

    monthly_samples = build_monthly_samples(
        monthly_features_df=monthly_features_df,
        reports_df=reports_df,
        incidence_df=incidence_df,
        report_embeddings_df=embeddings_df,
        stock_graph_edges_df=stock_edges_df,
        stock_clusters_df=stock_clusters_df,
        cluster_graph_edges_df=cluster_edges_df,
        decay_months=decay_months,
    )
    if signal not in monthly_samples:
        raise ValueError(f"Signal month sample not found: {signal}")
    train_months = select_training_months(
        monthly_samples,
        labels_df,
        signal_month=signal,
        train_start_month=train_start_month,
        train_end_month=train_end_month,
        training_label=training_label,
    )

    signal_sample = monthly_samples[signal]
    edge_type_mapping = dict(signal_sample.get("stock_graph_edge_type_to_id", {}))
    industry_mapping = _build_industry_mapping(monthly_features_df)
    checkpoint_dir = artifact_dir / "model_checkpoints" / "quarterly"
    canonical_checkpoint = checkpoint_path_for_training_end(
        checkpoint_dir,
        train_end_month=train_end_month,
        as_of_date=as_of_date,
    )
    expected = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "fusion_type": fusion_type,
        "graph_module_type": graph_module_type,
        "edge_type_prior_mode": edge_type_prior_mode,
        "training_objective": training_objective,
        "training_label": training_label,
        "train_start_month": _normalize_month(train_start_month),
        "train_end_month": _normalize_month(train_end_month),
        "train_months": train_months,
        "seed": int(seed),
        "epochs": int(epochs),
        "decay_months": int(decay_months),
        "community_reg_weight": float(community_reg_weight),
        "model_name": str(model_name),
        "edge_type_mapping": edge_type_mapping,
        "industry_mapping": industry_mapping,
    }
    if alpha_factor_manifest is not None:
        expected["alpha_factor_manifest"] = alpha_factor_manifest

    if canonical_checkpoint.exists() and not force_retrain:
        model, checkpoint_metadata = load_prediction_checkpoint(
            canonical_checkpoint,
            expected=expected,
        )
        checkpoint_path = canonical_checkpoint
        checkpoint_reused = True
    else:
        model = train_final_model(
            monthly_samples=monthly_samples,
            labels_df=labels_df,
            train_months=train_months,
            epochs=epochs,
            fusion_type=fusion_type,
            graph_module_type=graph_module_type,
            edge_type_prior_mode=edge_type_prior_mode,
            community_reg_weight=community_reg_weight,
            training_objective=training_objective,
            training_label=training_label,
            seed=seed,
        )
        checkpoint_metadata = build_checkpoint_metadata(
            model=model,
            train_months=train_months,
            train_start_month=train_start_month,
            train_end_month=train_end_month,
            as_of_date=as_of_date,
            signal_month=signal,
            seed=seed,
            fusion_type=fusion_type,
            graph_module_type=graph_module_type,
            edge_type_prior_mode=edge_type_prior_mode,
            training_objective=training_objective,
            training_label=training_label,
            edge_type_mapping=edge_type_mapping,
            industry_mapping=industry_mapping,
            alpha_factor_manifest=alpha_factor_manifest,
        )
        checkpoint_metadata.update(
            {
                "epochs": int(epochs),
                "decay_months": int(decay_months),
                "community_reg_weight": float(community_reg_weight),
                "checkpoint_cadence_months": int(checkpoint_cadence_months),
                "model_name": str(model_name),
            }
        )
        checkpoint_path = resolve_checkpoint_output_path(
            canonical_checkpoint,
            force_retrain=force_retrain,
            run_tag=run_tag.replace("-", "_").replace(":", ""),
        )
        save_prediction_checkpoint(model, checkpoint_metadata, checkpoint_path)
        checkpoint_reused = False

    scores_df = score_month(model, signal_sample)
    signal_features = monthly_features_df.loc[
        monthly_features_df["month"].astype(str).map(_normalize_month) == signal
    ].copy()
    recommendations_df = build_recommendations(
        scores_df,
        signal_features,
        stock_basic_df=stock_basic_df,
        topk=topk,
        liquidity_quantile=liquidity_quantile,
        min_price=min_price,
        exclude_st=exclude_st,
        as_of_date=as_of_date,
    )

    run_dir = artifact_dir / "prediction_runs" / run_tag
    run_dir.mkdir(parents=True, exist_ok=False)
    compact_signal = signal.replace("-", "")
    scores_path = run_dir / f"monthly_scores_{compact_signal}.parquet"
    recommendations_path = run_dir / f"recommendations_{compact_signal}.parquet"
    recommendations_csv = run_dir / f"recommendations_{compact_signal}.csv"
    metadata_path = run_dir / "prediction_metadata.json"
    scores_df.to_parquet(scores_path, index=False)
    recommendations_df.to_parquet(recommendations_path, index=False)
    recommendations_df.to_csv(recommendations_csv, index=False, encoding="utf-8-sig")

    prediction_metadata = {
        "run_tag": run_tag,
        "signal_month": signal,
        "as_of_date": pd.Timestamp(as_of_date).date().isoformat(),
        "execution_from": (
            pd.Timestamp(as_of_date) + pd.offsets.BDay(1)
        ).date().isoformat(),
        "train_start_month": _normalize_month(train_start_month),
        "train_end_month": _normalize_month(train_end_month),
        "train_months": train_months,
        "training_label": training_label,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_reused": checkpoint_reused,
        "score_rows": int(len(scores_df)),
        "recommendation_rows": int(len(recommendations_df)),
        "requested_topk": int(topk),
        "warning": (
            f"Only {len(recommendations_df)} eligible stocks were available for Top{topk}"
            if len(recommendations_df) < int(topk)
            else ""
        ),
        "checkpoint_metadata": checkpoint_metadata,
    }
    metadata_path.write_text(
        json.dumps(prediction_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "checkpoint": checkpoint_path,
        "run_dir": run_dir,
        "scores": scores_path,
        "recommendations": recommendations_path,
        "recommendations_csv": recommendations_csv,
        "metadata": metadata_path,
    }


def _build_industry_mapping(monthly_features_df: pd.DataFrame) -> dict[str, int]:
    if not {"industry", "industry_id"}.issubset(monthly_features_df.columns):
        return {}
    rows = monthly_features_df[["industry", "industry_id"]].dropna().drop_duplicates()
    return {
        str(row.industry): int(row.industry_id)
        for row in rows.sort_values(["industry_id", "industry"]).itertuples(index=False)
    }


def _read_required_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction artifact: {path}")
    return _read_frame(path)


def _read_optional_frame(path: Path) -> pd.DataFrame | None:
    return _read_frame(path) if path.exists() else None


def _read_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train or load a quarterly alphagraph checkpoint and score a signal month")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--signal-month", required=True)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--train-start-month", required=True)
    parser.add_argument("--train-end-month", required=True)
    parser.add_argument("--checkpoint-cadence-months", type=int, default=3)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--decay-months", type=int, default=3)
    parser.add_argument("--fusion-type", default="cross_encoder")
    parser.add_argument("--graph-module-type", default="relation_aware")
    parser.add_argument("--edge-type-prior-mode", default="none")
    parser.add_argument("--community-reg-weight", type=float, default=0.0)
    parser.add_argument("--training-objective", default="ranknet_pairwise")
    parser.add_argument("--training-label", default="future_excess_return_1m")
    parser.add_argument("--liquidity-quantile", type=float, default=0.2)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--exclude-st", type=_parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="")
    args = parser.parse_args(argv)
    paths = run_prediction(**vars(args))
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
