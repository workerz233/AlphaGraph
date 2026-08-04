from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


CLUSTER_STOCK_COLUMNS = [
    "month",
    "code",
    "cluster_id",
    "cluster_size",
    "cluster_score",
    "pagerank",
    "weighted_degree",
    "intra_cluster_degree",
    "report_attention",
    "top_industries",
    "cluster_method",
    "cluster_params_json",
]


def score_stock_clusters(
    month: str,
    graph: nx.Graph,
    month_features_df: pd.DataFrame,
    labels: dict[str, int],
    incidence_df: pd.DataFrame | None,
    cluster_method: str,
    cluster_params: dict[str, Any],
) -> pd.DataFrame:
    features = month_features_df.copy()
    features["code"] = features["code"].astype(str)
    features = features.set_index("code", drop=False)
    pagerank = pd.Series(nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {node: 1.0 / max(graph.number_of_nodes(), 1) for node in graph.nodes}, dtype=float)
    weighted_degree = pd.Series(dict(graph.degree(weight="weight")), dtype=float)
    report_attention = _report_attention(incidence_df).reindex(features.index).fillna(0.0)
    feature_score = _feature_score(features)

    row_frame = features[["code"]].copy()
    row_frame["cluster_id"] = [int(labels.get(code, index)) for index, code in enumerate(features.index)]
    row_frame["feature_score"] = feature_score
    row_frame["pagerank"] = pagerank.reindex(features.index).fillna(0.0)
    row_frame["weighted_degree"] = weighted_degree.reindex(features.index).fillna(0.0)
    row_frame["report_attention"] = report_attention
    row_frame["intra_cluster_degree"] = [
        _intra_cluster_degree(graph, code, labels) for code in features.index
    ]

    cluster_stats = []
    for cluster_id, group in row_frame.groupby("cluster_id"):
        feature_group = features.loc[group.index]
        top_industries = "/".join(
            f"{industry}:{count}"
            for industry, count in feature_group.get("industry", pd.Series(dtype=str)).fillna("").astype(str).value_counts().head(3).items()
        )
        cluster_stats.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(group)),
                "top_industries": top_industries,
                "mean_feature_score": float(group["feature_score"].mean()),
                "report_attention_sum": float(group["report_attention"].sum()),
            }
        )
    stats = pd.DataFrame(cluster_stats).set_index("cluster_id")
    stats["cluster_score"] = (
        0.70 * _zscore(stats["mean_feature_score"])
        + 0.15 * _zscore(np.log1p(stats["cluster_size"]))
        + 0.15 * _zscore(stats["report_attention_sum"])
    )

    row_frame["month"] = str(month)
    row_frame["cluster_size"] = stats["cluster_size"].reindex(row_frame["cluster_id"]).to_numpy()
    row_frame["cluster_score"] = stats["cluster_score"].reindex(row_frame["cluster_id"]).to_numpy()
    row_frame["top_industries"] = stats["top_industries"].reindex(row_frame["cluster_id"]).to_numpy()
    row_frame["cluster_method"] = str(cluster_method)
    row_frame["cluster_params_json"] = json.dumps(cluster_params, ensure_ascii=False, sort_keys=True)
    return row_frame.reset_index(drop=True).reindex(columns=CLUSTER_STOCK_COLUMNS)


def _feature_score(features: pd.DataFrame) -> pd.Series:
    return (
        0.22 * _zscore(features.get("ret_20d", 0.0))
        + 0.16 * _zscore(features.get("ret_60d", 0.0))
        + 0.16 * _zscore(features.get("rel_ret_20d_hs300", 0.0))
        + 0.10 * _zscore(features.get("rel_ret_60d_hs300", 0.0))
        + 0.12 * _zscore(np.log1p(pd.to_numeric(features.get("amount_20d_mean", 0.0), errors="coerce").fillna(0.0)))
        + 0.13 * _zscore(features.get("profit_roeAvg", 0.0))
        + 0.08 * _zscore(features.get("profit_npMargin", 0.0))
        + 0.10 * _zscore(features.get("growth_YOYNI", 0.0))
        - 0.07 * _zscore(features.get("vol_20d", 0.0))
    )


def _zscore(values: Any) -> pd.Series:
    series = pd.Series(values)
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = series.median(skipna=True)
    series = series.fillna(float(median) if pd.notna(median) else 0.0)
    if len(series) > 1:
        series = series.clip(series.quantile(0.02), series.quantile(0.98))
    std = series.std(ddof=0)
    if not std or math.isnan(float(std)):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _report_attention(incidence_df: pd.DataFrame | None) -> pd.Series:
    if incidence_df is None or incidence_df.empty or "stock_code" not in incidence_df.columns:
        return pd.Series(dtype=float)
    frame = incidence_df.copy()
    frame["code"] = frame["stock_code"].map(_normalize_stock_code)
    weights = pd.to_numeric(frame.get("incidence_weight", 1.0), errors="coerce").fillna(1.0)
    return weights.groupby(frame["code"]).sum()


def _normalize_stock_code(code: Any) -> str:
    raw = str(code).strip()
    if raw.startswith(("sh.", "sz.")):
        return raw
    if raw.endswith(".SH") or raw.endswith(".SZ"):
        return f"{raw[-2:].lower()}.{raw[:6]}"
    return raw


def _intra_cluster_degree(graph: nx.Graph, code: str, labels: dict[str, int]) -> float:
    cluster_id = labels.get(str(code))
    total = 0.0
    for neighbor, data in graph[str(code)].items() if str(code) in graph else []:
        if labels.get(str(neighbor)) == cluster_id:
            total += float(data.get("weight", 1.0))
    return total
