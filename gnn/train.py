from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from alphagraph.gnn.text_embedder import DEFAULT_EMBEDDING_MODEL_NAME
from alphagraph.model.core import (
    FEATURE_SCHEMA_VERSION,
    FUSION_MODULE_TYPES,
    GRAPH_MODULE_TYPES,
    TRAINING_LABELS,
    TRAINING_OBJECTIVES,
    build_monthly_samples,
    get_node_feature_columns,
    train_walk_forward,
)


ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2] / "artifacts" / "alphagraph"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train alphagraph selector")
    parser.add_argument(
        "--reports-path",
        type=Path,
        default=ARTIFACT_DIR / "reports.parquet",
    )
    parser.add_argument(
        "--incidence-path",
        type=Path,
        default=ARTIFACT_DIR / "incidence.parquet",
    )
    parser.add_argument(
        "--monthly-features-path",
        type=Path,
        default=ARTIFACT_DIR / "monthly_stock_features.parquet",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=ARTIFACT_DIR / "monthly_labels.parquet",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=ARTIFACT_DIR / "report_embeddings.parquet",
    )
    parser.add_argument(
        "--stock-graph-edges-path",
        type=Path,
        default=ARTIFACT_DIR / "stock_graph_edges.parquet",
    )
    parser.add_argument(
        "--stock-clusters-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--cluster-graph-edges-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=ARTIFACT_DIR / "monthly_scores.parquet",
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument("--decay-months", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--fusion-type",
        type=str,
        choices=FUSION_MODULE_TYPES,
        default="cross_encoder",
        help="Fusion module for ablation: concat, gated, or cross_encoder.",
    )
    parser.add_argument(
        "--graph-module-type",
        type=str,
        choices=GRAPH_MODULE_TYPES,
        default="mean",
        help="Stock graph aggregation module: mean or relation_aware.",
    )
    parser.add_argument(
        "--edge-type-prior-mode",
        type=str,
        choices=("none", "learnable"),
        default="none",
        help="Type-prior mode for relation_aware graph aggregation.",
    )
    parser.add_argument("--community-reg-weight", type=float, default=0.0)
    parser.add_argument(
        "--training-objective",
        type=str,
        choices=TRAINING_OBJECTIVES,
        default="ranknet_pairwise",
        help="Training objective: ranknet_pairwise for ranking or regression_mse for baseline.",
    )
    parser.add_argument(
        "--training-label",
        type=str,
        choices=TRAINING_LABELS,
        default="future_excess_return_1m",
        help="Training label column: return label or Sharpe-ratio label.",
    )
    args = parser.parse_args()

    reports_df = _read_required_frame(args.reports_path, "reports.parquet")
    incidence_df = _read_required_frame(args.incidence_path, "incidence.parquet")
    monthly_features_df = _read_required_frame(args.monthly_features_path, "monthly_stock_features.parquet")
    labels_df = _read_required_frame(args.labels_path, "monthly_labels.parquet")
    embeddings_df = _read_required_frame(args.embeddings_path, "report_embeddings.parquet")
    stock_graph_edges_df = _read_required_frame(args.stock_graph_edges_path, "stock_graph_edges.parquet")
    stock_clusters_df = _read_optional_frame(args.stock_clusters_path)
    cluster_graph_edges_df = _read_optional_frame(args.cluster_graph_edges_path)

    monthly_samples = build_monthly_samples(
        monthly_features_df=monthly_features_df,
        reports_df=reports_df,
        incidence_df=incidence_df,
        report_embeddings_df=embeddings_df,
        stock_graph_edges_df=stock_graph_edges_df,
        stock_clusters_df=stock_clusters_df,
        cluster_graph_edges_df=cluster_graph_edges_df,
        decay_months=args.decay_months,
    )
    scores_df = train_walk_forward(
        monthly_samples=monthly_samples,
        labels_df=labels_df,
        epochs=args.epochs,
        output_path=args.scores_path,
        fusion_type=args.fusion_type,
        graph_module_type=args.graph_module_type,
        edge_type_prior_mode=args.edge_type_prior_mode,
        community_reg_weight=args.community_reg_weight,
        training_objective=args.training_objective,
        training_label=args.training_label,
    )
    validate_training_scores(scores_df, monthly_samples)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ARTIFACT_DIR / "model_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            build_run_metadata(
                run_id=run_id,
                model_name=args.model_name,
                decay_months=args.decay_months,
                epochs=args.epochs,
                fusion_type=args.fusion_type,
                graph_module_type=args.graph_module_type,
                edge_type_prior_mode=args.edge_type_prior_mode,
                community_reg_weight=args.community_reg_weight,
                training_objective=args.training_objective,
                training_label=args.training_label,
                score_rows=int(len(scores_df)),
                alpha_factor_manifest=_read_alpha_manifest(args.monthly_features_path.parent),
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_run_metadata(
    run_id: str,
    model_name: str,
    decay_months: int,
    epochs: int,
    score_rows: int,
    fusion_type: str = "cross_encoder",
    graph_module_type: str = "mean",
    edge_type_prior_mode: str = "none",
    community_reg_weight: float = 0.0,
    training_objective: str = "ranknet_pairwise",
    training_label: str = "future_excess_return_1m",
    alpha_factor_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata = {
        "run_id": run_id,
        "model_name": model_name,
        "decay_months": decay_months,
        "epochs": epochs,
        "fusion_type": fusion_type,
        "graph_module_type": graph_module_type,
        "edge_type_prior_mode": edge_type_prior_mode,
        "community_reg_weight": float(community_reg_weight),
        "training_objective": str(training_objective),
        "training_label": str(training_label),
        "score_rows": int(score_rows),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "node_feature_columns": get_node_feature_columns(),
    }
    if alpha_factor_manifest is not None:
        metadata["alpha_factor_manifest"] = dict(alpha_factor_manifest)
    return metadata


def validate_training_scores(
    scores_df: pd.DataFrame,
    monthly_samples: dict[str, dict[str, object]],
) -> None:
    if not scores_df.empty:
        return
    months = sorted(str(month) for month in monthly_samples)
    month_text = ", ".join(months) if months else "无"
    raise RuntimeError(
        "训练未产生任何月度评分："
        f"有效图月份={len(months)}（{month_text}）。"
        "walk-forward 至少需要 2 个有效月份；请检查多月任务是否使用 markdown_root。"
    )


def _read_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)


def _read_required_frame(path: Path, artifact_name: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"缺少训练所需图产物 {artifact_name}: {path}。请先运行：bash alphagraph/build_kg.sh"
        )
    return _read_frame(path)


def _read_optional_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return _read_frame(path)


def _read_alpha_manifest(artifact_dir: Path) -> dict[str, object] | None:
    path = Path(artifact_dir) / "alpha_factor_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
