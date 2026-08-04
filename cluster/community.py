from __future__ import annotations

import pandas as pd

from alphagraph.cluster.graph_builder import build_cluster_graph_edges, build_month_graph
from alphagraph.cluster.registry import get_clusterer
from alphagraph.cluster.scoring import CLUSTER_STOCK_COLUMNS, score_stock_clusters


CLUSTER_EDGE_COLUMNS = [
    "month",
    "src_cluster_id",
    "dst_cluster_id",
    "edge_weight",
    "edge_count",
    "dominant_edge_type",
]


def build_stock_clusters(
    monthly_features_df: pd.DataFrame,
    stock_graph_edges_df: pd.DataFrame,
    incidence_df: pd.DataFrame | None = None,
    cluster_method: str = "louvain",
    cluster_resolution: float = 1.2,
    edge_type_weights: str | dict[str, float] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if monthly_features_df.empty:
        return (
            pd.DataFrame(columns=CLUSTER_STOCK_COLUMNS),
            pd.DataFrame(columns=CLUSTER_EDGE_COLUMNS),
        )
    features = monthly_features_df.copy()
    features["code"] = features["code"].astype(str)
    features["month"] = features["month"].astype(str)
    edges = stock_graph_edges_df.copy()
    if not edges.empty and "month" in edges.columns:
        edges["month"] = edges["month"].astype(str)

    clusterer = get_clusterer(cluster_method)
    cluster_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    params = clusterer.default_params()
    params.update({"resolution": float(cluster_resolution), "seed": int(seed)})
    if cluster_method not in {"leiden", "louvain"}:
        params.pop("resolution", None)

    for month, month_features in features.groupby("month", sort=True):
        graph = build_month_graph(
            month=str(month),
            month_features_df=month_features,
            stock_graph_edges_df=edges,
            edge_type_weights=edge_type_weights,
        )
        node_features = (
            month_features.set_index("code")
            .fillna("")
            .to_dict(orient="index")
        )
        labels = clusterer.fit_predict(graph, node_features, params)
        for index, code in enumerate(sorted(graph.nodes)):
            labels.setdefault(str(code), index)
        cluster_frames.append(
            score_stock_clusters(
                month=str(month),
                graph=graph,
                month_features_df=month_features,
                labels=labels,
                incidence_df=incidence_df,
                cluster_method=clusterer.name,
                cluster_params=params,
            )
        )
        edge_frames.append(build_cluster_graph_edges(str(month), graph, labels))

    clusters = pd.concat(cluster_frames, ignore_index=True) if cluster_frames else pd.DataFrame(columns=CLUSTER_STOCK_COLUMNS)
    cluster_edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame(columns=CLUSTER_EDGE_COLUMNS)
    return clusters.reindex(columns=CLUSTER_STOCK_COLUMNS), cluster_edges.reindex(columns=CLUSTER_EDGE_COLUMNS)
