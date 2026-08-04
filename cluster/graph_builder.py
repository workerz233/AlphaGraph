from __future__ import annotations

from collections import Counter, defaultdict

import networkx as nx
import pandas as pd


DEFAULT_EDGE_TYPE_WEIGHTS = {
    "market_corr_topk": 0.45,
    "same_industry_topk": 0.70,
    "same_industry": 1.20,
    "text_similarity": 1.35,
    "kg_llm": 1.50,
}


def parse_edge_type_weights(raw: str | dict[str, float] | None) -> dict[str, float]:
    if raw is None or raw == "":
        return dict(DEFAULT_EDGE_TYPE_WEIGHTS)
    if isinstance(raw, dict):
        weights = dict(DEFAULT_EDGE_TYPE_WEIGHTS)
        weights.update({str(key): float(value) for key, value in raw.items()})
        return weights
    weights = dict(DEFAULT_EDGE_TYPE_WEIGHTS)
    for part in str(raw).split(","):
        if not part.strip():
            continue
        key, sep, value = part.partition(":")
        if not sep:
            raise ValueError(f"Invalid edge type weight entry: {part!r}")
        weights[key.strip()] = float(value)
    return weights


def build_month_graph(
    month: str,
    month_features_df: pd.DataFrame,
    stock_graph_edges_df: pd.DataFrame,
    edge_type_weights: str | dict[str, float] | None = None,
) -> nx.Graph:
    weights = parse_edge_type_weights(edge_type_weights)
    graph = nx.Graph()
    for code in month_features_df["code"].astype(str):
        graph.add_node(code)
    if stock_graph_edges_df.empty:
        return graph
    edges = stock_graph_edges_df.loc[stock_graph_edges_df["month"].astype(str).eq(str(month))].copy()
    edge_acc: defaultdict[tuple[str, str], float] = defaultdict(float)
    edge_types: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in edges.itertuples(index=False):
        src = str(row.src_code)
        dst = str(row.dst_code)
        if src == dst or src not in graph or dst not in graph:
            continue
        key = tuple(sorted((src, dst)))
        edge_type = str(getattr(row, "edge_type", ""))
        edge_weight = float(getattr(row, "edge_weight", 1.0) or 1.0)
        edge_acc[key] += edge_weight * weights.get(edge_type, 1.0)
        edge_types[key][edge_type] += 1
    for (src, dst), weight in edge_acc.items():
        graph.add_edge(src, dst, weight=float(weight), edge_types=dict(edge_types[(src, dst)]))
    return graph


def build_cluster_graph_edges(
    month: str,
    graph: nx.Graph,
    labels: dict[str, int],
) -> pd.DataFrame:
    rows_by_pair: dict[tuple[int, int], dict[str, object]] = {}
    edge_type_counts: defaultdict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for src, dst, data in graph.edges(data=True):
        src_cluster = int(labels.get(str(src), -1))
        dst_cluster = int(labels.get(str(dst), -1))
        if src_cluster < 0 or dst_cluster < 0 or src_cluster == dst_cluster:
            continue
        key = tuple(sorted((src_cluster, dst_cluster)))
        row = rows_by_pair.setdefault(
            key,
            {
                "month": str(month),
                "src_cluster_id": key[0],
                "dst_cluster_id": key[1],
                "edge_weight": 0.0,
                "edge_count": 0,
                "dominant_edge_type": "",
            },
        )
        row["edge_weight"] = float(row["edge_weight"]) + float(data.get("weight", 1.0))
        row["edge_count"] = int(row["edge_count"]) + 1
        for edge_type, count in data.get("edge_types", {}).items():
            edge_type_counts[key][str(edge_type)] += int(count)
    for key, row in rows_by_pair.items():
        if edge_type_counts[key]:
            row["dominant_edge_type"] = edge_type_counts[key].most_common(1)[0][0]
    return pd.DataFrame(
        rows_by_pair.values(),
        columns=[
            "month",
            "src_cluster_id",
            "dst_cluster_id",
            "edge_weight",
            "edge_count",
            "dominant_edge_type",
        ],
    )
