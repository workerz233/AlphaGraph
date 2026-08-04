from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alphagraph.cluster.community import build_stock_clusters


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "alphagraph"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pluggable alphagraph stock clusters")
    parser.add_argument("--monthly-features-path", type=Path, required=True)
    parser.add_argument("--stock-graph-edges-path", type=Path, required=True)
    parser.add_argument("--incidence-path", type=Path, default=None)
    parser.add_argument("--output-clusters", type=Path, required=True)
    parser.add_argument("--output-cluster-edges", type=Path, required=True)
    parser.add_argument("--cluster-method", type=str, default="louvain")
    parser.add_argument("--cluster-resolution", type=float, default=1.2)
    parser.add_argument("--cluster-edge-type-weights", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    monthly_features_df = _read_frame(args.monthly_features_path)
    stock_graph_edges_df = _read_frame(args.stock_graph_edges_path)
    incidence_df = (
        _read_frame(args.incidence_path)
        if args.incidence_path is not None and args.incidence_path.exists()
        else pd.DataFrame()
    )
    clusters, cluster_edges = build_stock_clusters(
        monthly_features_df=monthly_features_df,
        stock_graph_edges_df=stock_graph_edges_df,
        incidence_df=incidence_df,
        cluster_method=args.cluster_method,
        cluster_resolution=args.cluster_resolution,
        edge_type_weights=args.cluster_edge_type_weights,
        seed=args.seed,
    )
    _write_frame(clusters, args.output_clusters)
    _write_frame(cluster_edges, args.output_cluster_edges)


def _read_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
    except Exception:
        frame.to_pickle(path)


if __name__ == "__main__":
    main()
