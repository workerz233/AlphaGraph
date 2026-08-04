from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from alphagraph.data.market_data import normalize_baostock_code


ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2] / "artifacts" / "alphagraph"
)
FEATURE_SCHEMA_VERSION = "stock-node-v3-alpha10"
NODE_FEATURE_COLUMNS = [
    "ret_20d",
    "ret_60d",
    "vol_20d",
    "vol_60d",
    "amount_20d_mean",
    "turn_20d_mean",
    "rel_ret_20d_hs300",
    "pct_chg_20d_mean",
    "pct_chg_20d_std",
    "pe_ttm",
    "pb_mrq",
    "ps_ttm",
    "pcf_ncf_ttm",
    "is_st",
    "listing_age_days",
    "security_type_id",
    "is_hs300",
    "is_zz500",
    "is_sz50",
    "profit_roeAvg",
    "profit_npMargin",
    "profit_gpMargin",
    "growth_YOYNI",
    "growth_YOYAsset",
    "operation_NRTurnRatio",
    "operation_INVTurnRatio",
    "balance_currentRatio",
    "balance_quickRatio",
    "dupont_dupontROE",
    "financial_report_age_days",
    "has_financial_features",
    "alpha_A191_alpha074",
    "alpha_A101_alpha047",
    "alpha_A191_alpha179",
    "alpha_A158_HIGH0",
    "alpha_A101_alpha045",
    "alpha_A191_alpha113",
    "alpha_A101_alpha012",
    "alpha_A158_IMIN10",
    "alpha_A101_alpha022",
    "alpha_A191_alpha104",
]
FUSION_MODULE_TYPES = ("concat", "gated", "cross_encoder")
GRAPH_MODULE_TYPES = ("mean", "relation_aware", "community_aware")
STOCK_EDGE_BUILDERS = ("report_co_coverage", "market_corr_topk", "same_industry_topk", "kg_llm")
TRAINING_OBJECTIVES = ("ranknet_pairwise", "regression_mse")
TRAINING_LABELS = ("future_excess_return_1m", "future_excess_sharpe_1m")


def get_node_feature_columns() -> list[str]:
    return list(NODE_FEATURE_COLUMNS)


def build_stock_graph_edges(
    monthly_features_df: pd.DataFrame,
    reports_df: pd.DataFrame,
    incidence_df: pd.DataFrame,
    report_embeddings_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    del monthly_features_df

    normalized_incidence = incidence_df.copy()
    if not normalized_incidence.empty and "stock_code" in normalized_incidence.columns:
        normalized_incidence["stock_code"] = normalized_incidence["stock_code"].map(
            _normalize_stock_code
        )

    report_meta = reports_df.set_index("report_id").to_dict(orient="index")
    embedding_lookup = (
        report_embeddings_df.set_index("report_id").filter(like="emb_").to_dict(orient="index")
        if report_embeddings_df is not None and not report_embeddings_df.empty
        else {}
    )

    edges: list[dict[str, Any]] = []
    for report_id, report_rows in normalized_incidence.groupby("report_id"):
        codes = sorted(code for code in report_rows["stock_code"].unique().tolist() if code)
        if len(codes) < 2:
            continue
        industry = str(report_meta.get(report_id, {}).get("industry", "")).strip()
        report_month = str(report_meta.get(report_id, {}).get("report_month", "")).strip()
        has_embedding = report_id in embedding_lookup

        for src_code, dst_code in combinations(codes, 2):
            if industry:
                edges.append(
                    {
                        "month": report_month,
                        "src_code": src_code,
                        "dst_code": dst_code,
                        "edge_type": "same_industry",
                        "edge_weight": 1.0,
                        "source_id": report_id,
                        "report_id": report_id,
                    }
                )
            if has_embedding:
                vector = np.array(list(embedding_lookup[report_id].values()), dtype=float)
                similarity = float(np.linalg.norm(vector)) if vector.size else 0.0
                edges.append(
                    {
                        "month": report_month,
                        "src_code": src_code,
                        "dst_code": dst_code,
                        "edge_type": "text_similarity",
                        "edge_weight": similarity,
                        "source_id": report_id,
                        "report_id": report_id,
                    }
                )

    return pd.DataFrame(
        edges,
        columns=["month", "src_code", "dst_code", "edge_type", "edge_weight", "source_id", "report_id"],
    )


def build_configured_stock_graph_edges(
    monthly_features_df: pd.DataFrame,
    reports_df: pd.DataFrame,
    incidence_df: pd.DataFrame,
    report_embeddings_df: pd.DataFrame | None = None,
    daily_k_df: pd.DataFrame | None = None,
    hs300_daily_df: pd.DataFrame | None = None,
    edge_builders: str | list[str] = "report_co_coverage",
    market_corr_lookback_days: int = 60,
    market_corr_topk: int = 10,
    market_corr_min_score: float = 0.1,
    industry_topk: int = 5,
    llm_relations_df: pd.DataFrame | None = None,
    llm_kg_decay_months: int = 3,
) -> pd.DataFrame:
    builders = _parse_stock_edge_builders(edge_builders)
    frames: list[pd.DataFrame] = []
    if "report_co_coverage" in builders:
        frames.append(
            build_stock_graph_edges(
                monthly_features_df=monthly_features_df,
                reports_df=reports_df,
                incidence_df=incidence_df,
                report_embeddings_df=report_embeddings_df,
            )
        )
    if "market_corr_topk" in builders:
        frames.append(
            build_market_corr_topk_edges(
                monthly_features_df=monthly_features_df,
                daily_k_df=daily_k_df if daily_k_df is not None else pd.DataFrame(),
                hs300_daily_df=hs300_daily_df,
                lookback_days=market_corr_lookback_days,
                topk=market_corr_topk,
                min_score=market_corr_min_score,
            )
        )
    if "same_industry_topk" in builders:
        frames.append(
            build_same_industry_topk_edges(
                monthly_features_df=monthly_features_df,
                daily_k_df=daily_k_df if daily_k_df is not None else pd.DataFrame(),
                hs300_daily_df=hs300_daily_df,
                lookback_days=market_corr_lookback_days,
                topk=industry_topk,
                min_score=market_corr_min_score,
            )
        )
    if "kg_llm" in builders:
        frames.append(
            build_kg_llm_stock_edges(
                monthly_features_df=monthly_features_df,
                llm_relations_df=llm_relations_df if llm_relations_df is not None else pd.DataFrame(),
                llm_kg_decay_months=llm_kg_decay_months,
            )
        )
    if not frames:
        return _empty_stock_edges()
    combined = _normalize_stock_edge_frame(pd.concat(frames, ignore_index=True))
    if combined.empty:
        return _empty_stock_edges()
    return combined.drop_duplicates(
        subset=["month", "src_code", "dst_code", "edge_type", "source_id"],
        keep="last",
    ).reset_index(drop=True)


def build_market_corr_topk_edges(
    monthly_features_df: pd.DataFrame,
    daily_k_df: pd.DataFrame,
    hs300_daily_df: pd.DataFrame | None = None,
    lookback_days: int = 60,
    topk: int = 10,
    min_score: float = 0.1,
) -> pd.DataFrame:
    return _build_topk_similarity_edges(
        monthly_features_df=monthly_features_df,
        daily_k_df=daily_k_df,
        hs300_daily_df=hs300_daily_df,
        lookback_days=lookback_days,
        topk=topk,
        min_score=min_score,
        edge_type="market_corr_topk",
        same_industry_only=False,
    )


def build_same_industry_topk_edges(
    monthly_features_df: pd.DataFrame,
    daily_k_df: pd.DataFrame,
    hs300_daily_df: pd.DataFrame | None = None,
    lookback_days: int = 60,
    topk: int = 5,
    min_score: float = 0.1,
) -> pd.DataFrame:
    return _build_topk_similarity_edges(
        monthly_features_df=monthly_features_df,
        daily_k_df=daily_k_df,
        hs300_daily_df=hs300_daily_df,
        lookback_days=lookback_days,
        topk=topk,
        min_score=min_score,
        edge_type="same_industry_topk",
        same_industry_only=True,
    )


def build_kg_llm_stock_edges(
    monthly_features_df: pd.DataFrame,
    llm_relations_df: pd.DataFrame,
    llm_kg_decay_months: int = 3,
) -> pd.DataFrame:
    if monthly_features_df.empty or llm_relations_df.empty:
        return _empty_stock_edges()
    monthly = monthly_features_df.copy()
    monthly["code"] = monthly["code"].map(_normalize_stock_code)
    relations = llm_relations_df.copy()
    relations["report_month"] = relations.get("report_month", "").fillna("").astype(str)
    relations["available_month"] = _available_month_series(relations)
    relations["relation_type"] = relations.get("relation_type", "").fillna("").astype(str)
    relations["head_code"] = relations.get("head_code", "").map(_normalize_stock_code)
    relations["tail_code"] = relations.get("tail_code", "").map(_normalize_stock_code)
    relations["confidence"] = pd.to_numeric(relations.get("confidence", 0.0), errors="coerce").fillna(0.0)
    for column in ("topic", "driver_type", "risk_type", "source_section", "report_id"):
        if column not in relations.columns:
            relations[column] = ""
        relations[column] = relations[column].fillna("").astype(str)

    rows: list[dict[str, Any]] = []
    for month, month_rows in monthly.groupby("month"):
        stock_codes = set(month_rows["code"].dropna().astype(str).tolist())
        active_relations = relations.loc[
            relations["available_month"].map(
                lambda available_month: _is_active_report_month(
                    available_month,
                    str(month),
                    max(int(llm_kg_decay_months), 1),
                )
            )
        ].copy()
        if active_relations.empty:
            continue
        rows.extend(
            _llm_stock_pair_edges(
                active_relations,
                stock_codes,
                str(month),
                llm_kg_decay_months,
            )
        )
    if not rows:
        return _empty_stock_edges()
    return pd.DataFrame(rows, columns=_empty_stock_edges().columns)


def build_monthly_samples(
    monthly_features_df: pd.DataFrame,
    reports_df: pd.DataFrame,
    incidence_df: pd.DataFrame,
    report_embeddings_df: pd.DataFrame,
    stock_graph_edges_df: pd.DataFrame,
    stock_clusters_df: pd.DataFrame | None = None,
    cluster_graph_edges_df: pd.DataFrame | None = None,
    decay_months: int = 3,
) -> dict[str, dict[str, Any]]:
    normalized_features = monthly_features_df.copy()
    if "code" in normalized_features.columns:
        normalized_features["code"] = normalized_features["code"].map(_normalize_stock_code)

    normalized_incidence = incidence_df.copy()
    if "stock_code" in normalized_incidence.columns:
        normalized_incidence["stock_code"] = normalized_incidence["stock_code"].map(
            _normalize_stock_code
        )

    normalized_stock_edges = stock_graph_edges_df.copy()
    if normalized_stock_edges.empty:
        normalized_stock_edges = _empty_stock_edges()
    if "src_code" in normalized_stock_edges.columns:
        normalized_stock_edges["src_code"] = normalized_stock_edges["src_code"].map(
            _normalize_stock_code
        )
    if "dst_code" in normalized_stock_edges.columns:
        normalized_stock_edges["dst_code"] = normalized_stock_edges["dst_code"].map(
            _normalize_stock_code
        )
    if "month" not in normalized_stock_edges.columns:
        normalized_stock_edges["month"] = ""
    if "report_id" not in normalized_stock_edges.columns:
        normalized_stock_edges["report_id"] = ""
    normalized_clusters = stock_clusters_df.copy() if stock_clusters_df is not None else pd.DataFrame()
    if not normalized_clusters.empty:
        normalized_clusters["code"] = normalized_clusters["code"].map(_normalize_stock_code)
        normalized_clusters["month"] = normalized_clusters["month"].astype(str)
    normalized_cluster_edges = (
        cluster_graph_edges_df.copy() if cluster_graph_edges_df is not None else pd.DataFrame()
    )
    if not normalized_cluster_edges.empty:
        normalized_cluster_edges["month"] = normalized_cluster_edges["month"].astype(str)

    months = sorted(normalized_features["month"].unique())
    report_embeddings = (
        report_embeddings_df.set_index("report_id").filter(like="emb_")
        if not report_embeddings_df.empty
        else pd.DataFrame()
    )
    samples: dict[str, dict[str, Any]] = {}

    for month in months:
        month_rows = normalized_features.loc[normalized_features["month"] == month].copy()
        month_rows = month_rows.sort_values("code").reset_index(drop=True)
        stock_codes = month_rows["code"].tolist()
        stock_index = {code: idx for idx, code in enumerate(stock_codes)}

        report_available_months = _available_month_series(reports_df)
        active_reports = reports_df.loc[
            report_available_months.map(
                lambda available_month: _is_active_report_month(
                    report_month=available_month,
                    target_month=month,
                    decay_months=decay_months,
                )
            )
        ].copy()
        active_reports = active_reports.sort_values("report_id").reset_index(drop=True)
        report_ids = active_reports["report_id"].tolist()
        report_index = {report_id: idx for idx, report_id in enumerate(report_ids)}

        hyperedge_features = (
            report_embeddings.reindex(report_ids).fillna(0.0).to_numpy(dtype=np.float32)
            if report_ids
            else np.zeros((0, int(report_embeddings.shape[1])), dtype=np.float32)
        )

        incidence_rows = normalized_incidence.loc[
            normalized_incidence["report_id"].isin(report_ids)
            & normalized_incidence["stock_code"].isin(stock_codes)
        ].copy()
        incidence_rows = incidence_rows.sort_values(["report_id", "stock_code"]).reset_index(
            drop=True
        )

        incidence_index = np.array(
            [
                [stock_index[row.stock_code] for row in incidence_rows.itertuples()],
                [report_index[row.report_id] for row in incidence_rows.itertuples()],
            ],
            dtype=np.int64,
        )
        if incidence_rows.empty:
            incidence_index = np.zeros((2, 0), dtype=np.int64)

        edge_type_to_id = {
            edge_type: edge_type_id
            for edge_type_id, edge_type in enumerate(
                sorted(normalized_stock_edges.get("edge_type", pd.Series(dtype=str)).dropna().unique().tolist())
            )
        }

        report_edge_mask = normalized_stock_edges["report_id"].isin(report_ids)
        month_edge_mask = normalized_stock_edges["month"].astype(str).eq(str(month))
        legacy_edge_mask = normalized_stock_edges["month"].astype(str).eq("") & report_edge_mask
        rolling_report_edge_mask = report_edge_mask & _is_rolling_report_edge_type(
            normalized_stock_edges.get("edge_type", pd.Series(dtype=str))
        )
        stock_edges = normalized_stock_edges.loc[
            normalized_stock_edges["src_code"].isin(stock_codes)
            & normalized_stock_edges["dst_code"].isin(stock_codes)
            & (month_edge_mask | legacy_edge_mask | rolling_report_edge_mask)
        ].copy()
        stock_edges = stock_edges.sort_values(["edge_type", "src_code", "dst_code"]).reset_index(
            drop=True
        )
        stock_graph_edges = np.array(
            [
                [stock_index[row.src_code] for row in stock_edges.itertuples()],
                [stock_index[row.dst_code] for row in stock_edges.itertuples()],
            ],
            dtype=np.int64,
        )
        if stock_edges.empty:
            stock_graph_edges = np.zeros((2, 0), dtype=np.int64)
        stock_graph_edge_type_ids = (
            stock_edges["edge_type"].map(edge_type_to_id).fillna(0).to_numpy(dtype=np.int64)
            if not stock_edges.empty
            else np.zeros((0,), dtype=np.int64)
        )
        stock_graph_edge_weights = (
            stock_edges.get("edge_weight", pd.Series([1.0] * len(stock_edges)))
            .fillna(1.0)
            .to_numpy(dtype=np.float32)
            if not stock_edges.empty
            else np.zeros((0,), dtype=np.float32)
        )
        month_clusters = normalized_clusters.loc[
            normalized_clusters.get("month", pd.Series(dtype=str)).astype(str).eq(str(month))
            & normalized_clusters.get("code", pd.Series(dtype=str)).isin(stock_codes)
        ].copy()
        stock_cluster_index, cluster_features, cluster_graph_edges, cluster_graph_edge_weights = (
            _build_cluster_sample_arrays(
                month=str(month),
                stock_codes=stock_codes,
                month_clusters=month_clusters,
                cluster_edges_df=normalized_cluster_edges,
            )
        )

        node_feature_frame = month_rows.reindex(columns=NODE_FEATURE_COLUMNS)
        node_feature_frame = node_feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        samples[month] = {
            "month": month,
            "stock_codes": stock_codes,
            "node_features": node_feature_frame.to_numpy(dtype=np.float32),
            "industry_id": month_rows.get(
                "industry_id", pd.Series([0] * len(month_rows))
            ).fillna(0).to_numpy(dtype=np.int64),
            "hyperedge_features": hyperedge_features,
            "incidence_index": incidence_index,
            "incidence_weight": incidence_rows.get(
                "incidence_weight", pd.Series(dtype=float)
            ).to_numpy(dtype=np.float32),
            "stock_graph_edges": stock_graph_edges,
            "stock_graph_edge_type_ids": stock_graph_edge_type_ids,
            "stock_graph_edge_weights": stock_graph_edge_weights,
            "stock_graph_edge_types": sorted(stock_edges["edge_type"].unique().tolist()),
            "stock_graph_edge_type_to_id": edge_type_to_id,
            "stock_cluster_index": stock_cluster_index,
            "cluster_features": cluster_features,
            "cluster_graph_edges": cluster_graph_edges,
            "cluster_graph_edge_weights": cluster_graph_edge_weights,
        }

    return samples


class DualChannelSelector(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        hyperedge_feature_dim: int,
        hidden_dim: int = 32,
        num_industries: int = 1,
        fusion_type: str = "cross_encoder",
        graph_module_type: str = "mean",
        num_edge_types: int = 1,
        edge_type_prior_mode: str = "none",
    ) -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.graph_module_type = graph_module_type
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.industry_embedding = nn.Embedding(max(num_industries, 1), hidden_dim)
        self.hyperedge_encoder = nn.Linear(max(hyperedge_feature_dim, 1), hidden_dim)
        self.stock_graph_module = build_stock_graph_module(
            graph_module_type=graph_module_type,
            hidden_dim=hidden_dim,
            num_edge_types=num_edge_types,
            edge_type_prior_mode=edge_type_prior_mode,
        )
        self.community_graph_module = (
            CommunityAwareStockGraphModule(hidden_dim)
            if str(graph_module_type).strip().lower() == "community_aware"
            else None
        )
        self.fusion_module = build_fusion_module(fusion_type, hidden_dim)

    def forward(self, sample: dict[str, Any]) -> torch.Tensor:
        node_features = sample["node_features"]
        hyperedge_features = sample["hyperedge_features"]
        incidence_index = sample["incidence_index"]
        incidence_weight = sample.get("incidence_weight")
        stock_graph_edges = sample["stock_graph_edges"]
        stock_graph_edge_type_ids = sample.get("stock_graph_edge_type_ids")
        stock_graph_edge_weights = sample.get("stock_graph_edge_weights")
        stock_cluster_index = sample.get("stock_cluster_index")
        cluster_features = sample.get("cluster_features")
        cluster_graph_edges = sample.get("cluster_graph_edges")
        cluster_graph_edge_weights = sample.get("cluster_graph_edge_weights")

        node_hidden = self.node_encoder(node_features)
        industry_id = sample.get("industry_id")
        if industry_id is not None:
            industry_id = industry_id.clamp(min=0, max=self.industry_embedding.num_embeddings - 1)
            node_hidden = node_hidden + self.industry_embedding(industry_id)
        hyper_hidden = self.hyperedge_encoder(_ensure_non_empty_feature_matrix(hyperedge_features))
        hyper_message = self._aggregate_hyperedges(
            num_nodes=node_hidden.shape[0],
            hidden_dim=node_hidden.shape[1],
            hyper_hidden=hyper_hidden,
            incidence_index=incidence_index,
            incidence_weight=incidence_weight,
            device=node_hidden.device,
        )
        stock_message = self.stock_graph_module(
            node_hidden,
            stock_graph_edges,
            stock_graph_edge_type_ids,
            stock_graph_edge_weights,
        )
        if self.community_graph_module is not None:
            community_message = self.community_graph_module(
                node_hidden=node_hidden,
                stock_cluster_index=stock_cluster_index,
                cluster_features=cluster_features,
                cluster_graph_edges=cluster_graph_edges,
                cluster_graph_edge_weights=cluster_graph_edge_weights,
            )
            stock_message = stock_message + community_message
        return self.fusion_module(node_hidden, hyper_message, stock_message)

    def _aggregate_hyperedges(
        self,
        num_nodes: int,
        hidden_dim: int,
        hyper_hidden: torch.Tensor,
        incidence_index: torch.Tensor,
        incidence_weight: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if incidence_index.numel() == 0 or hyper_hidden.numel() == 0:
            return torch.zeros((num_nodes, hidden_dim), device=device)

        stock_idx = incidence_index[0]
        report_idx = incidence_index[1]
        weights = (
            incidence_weight
            if incidence_weight is not None
            else torch.ones(report_idx.shape[0], dtype=hyper_hidden.dtype, device=device)
        )
        messages = hyper_hidden[report_idx] * weights.unsqueeze(-1)
        aggregated = torch.zeros((num_nodes, hidden_dim), dtype=hyper_hidden.dtype, device=device)
        counts = torch.zeros((num_nodes, 1), dtype=hyper_hidden.dtype, device=device)
        aggregated.index_add_(0, stock_idx, messages)
        counts.index_add_(0, stock_idx, weights.unsqueeze(-1))
        counts = counts.clamp_min(1.0)
        return aggregated / counts

class MeanStockGraphModule(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.stock_graph_encoder = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        node_hidden: torch.Tensor,
        stock_graph_edges: torch.Tensor,
        stock_graph_edge_type_ids: torch.Tensor | None = None,
        stock_graph_edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del stock_graph_edge_type_ids, stock_graph_edge_weights
        num_nodes, hidden_dim = node_hidden.shape
        if stock_graph_edges.numel() == 0:
            return torch.zeros((num_nodes, hidden_dim), dtype=node_hidden.dtype, device=node_hidden.device)

        src = stock_graph_edges[0]
        dst = stock_graph_edges[1]
        neighbor_messages = self.stock_graph_encoder(node_hidden[src])
        aggregated = torch.zeros_like(node_hidden)
        counts = torch.zeros((num_nodes, 1), dtype=node_hidden.dtype, device=node_hidden.device)
        aggregated.index_add_(0, dst, neighbor_messages)
        counts.index_add_(
            0,
            dst,
            torch.ones((dst.shape[0], 1), dtype=node_hidden.dtype, device=node_hidden.device),
        )
        counts = counts.clamp_min(1.0)
        return aggregated / counts


class RelationAwareStockGraphModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_edge_types: int = 1,
        edge_type_prior_mode: str = "none",
    ) -> None:
        super().__init__()
        edge_type_prior_mode = str(edge_type_prior_mode).strip().lower()
        if edge_type_prior_mode not in {"none", "learnable"}:
            raise ValueError(
                "Unsupported edge_type_prior_mode="
                f"{edge_type_prior_mode!r}; expected one of ('none', 'learnable')"
            )
        self.num_edge_types = max(num_edge_types, 1)
        self.edge_type_prior_mode = edge_type_prior_mode
        self.edge_type_embedding = nn.Embedding(self.num_edge_types, hidden_dim)
        if self.edge_type_prior_mode == "learnable":
            self.edge_type_prior_logits = nn.Parameter(torch.zeros(self.num_edge_types))
        self.message_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_hidden: torch.Tensor,
        stock_graph_edges: torch.Tensor,
        stock_graph_edge_type_ids: torch.Tensor | None = None,
        stock_graph_edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_nodes, hidden_dim = node_hidden.shape
        if stock_graph_edges.numel() == 0:
            return torch.zeros((num_nodes, hidden_dim), dtype=node_hidden.dtype, device=node_hidden.device)

        src = stock_graph_edges[0]
        dst = stock_graph_edges[1]
        edge_type_ids = _normalize_edge_type_ids(
            stock_graph_edge_type_ids,
            num_edges=src.shape[0],
            max_edge_type_id=self.num_edge_types - 1,
            device=node_hidden.device,
        )
        edge_weights = _normalize_edge_weights(
            stock_graph_edge_weights,
            num_edges=src.shape[0],
            dtype=node_hidden.dtype,
            device=node_hidden.device,
        )
        effective_edge_weights = self._apply_edge_type_prior(edge_weights, edge_type_ids)

        relation_hidden = self.edge_type_embedding(edge_type_ids)
        source_message = self.message_encoder(node_hidden[src]) + relation_hidden
        edge_weight_feature = effective_edge_weights.clamp_min(0.0).unsqueeze(-1)
        attention_input = torch.cat(
            [node_hidden[src], node_hidden[dst], source_message, edge_weight_feature],
            dim=-1,
        )
        attention_logits = self.attention(attention_input).squeeze(-1)
        attention_logits = attention_logits + torch.log(effective_edge_weights.clamp_min(1e-6))
        attention_weights = _edge_softmax(attention_logits, dst, num_nodes)

        aggregated = torch.zeros((num_nodes, hidden_dim), dtype=node_hidden.dtype, device=node_hidden.device)
        aggregated.index_add_(0, dst, source_message * attention_weights.unsqueeze(-1))
        return aggregated

    def _apply_edge_type_prior(
        self,
        edge_weights: torch.Tensor,
        edge_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.edge_type_prior_mode != "learnable":
            return edge_weights
        type_prior = torch.nn.functional.softplus(self.edge_type_prior_logits[edge_type_ids])
        neutral_prior = torch.nn.functional.softplus(edge_weights.new_tensor(0.0))
        return edge_weights * (type_prior / neutral_prior)


class CommunityAwareStockGraphModule(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.cluster_feature_encoder = nn.Linear(5, hidden_dim)
        self.cluster_message_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.stock_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        node_hidden: torch.Tensor,
        stock_cluster_index: torch.Tensor | None,
        cluster_features: torch.Tensor | None,
        cluster_graph_edges: torch.Tensor | None,
        cluster_graph_edge_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        num_nodes, hidden_dim = node_hidden.shape
        if (
            stock_cluster_index is None
            or cluster_features is None
            or stock_cluster_index.numel() == 0
            or cluster_features.numel() == 0
        ):
            return torch.zeros((num_nodes, hidden_dim), dtype=node_hidden.dtype, device=node_hidden.device)
        cluster_index = stock_cluster_index.to(device=node_hidden.device, dtype=torch.long).clamp_min(0)
        num_clusters = int(cluster_features.shape[0])
        if num_clusters <= 0:
            return torch.zeros((num_nodes, hidden_dim), dtype=node_hidden.dtype, device=node_hidden.device)
        cluster_index = cluster_index.clamp(max=num_clusters - 1)
        cluster_features = cluster_features.to(device=node_hidden.device, dtype=node_hidden.dtype)
        if cluster_features.shape[1] < 5:
            pad = torch.zeros(
                (cluster_features.shape[0], 5 - cluster_features.shape[1]),
                dtype=node_hidden.dtype,
                device=node_hidden.device,
            )
            cluster_features = torch.cat([cluster_features, pad], dim=1)
        cluster_hidden = self.cluster_feature_encoder(cluster_features[:, :5])
        aggregated = torch.zeros((num_clusters, hidden_dim), dtype=node_hidden.dtype, device=node_hidden.device)
        counts = torch.zeros((num_clusters, 1), dtype=node_hidden.dtype, device=node_hidden.device)
        aggregated.index_add_(0, cluster_index, node_hidden)
        counts.index_add_(
            0,
            cluster_index,
            torch.ones((num_nodes, 1), dtype=node_hidden.dtype, device=node_hidden.device),
        )
        cluster_hidden = cluster_hidden + aggregated / counts.clamp_min(1.0)
        cluster_hidden = self._propagate_cluster_graph(
            cluster_hidden,
            cluster_graph_edges,
            cluster_graph_edge_weights,
        )
        return self.stock_update(torch.cat([node_hidden, cluster_hidden[cluster_index]], dim=-1))

    def _propagate_cluster_graph(
        self,
        cluster_hidden: torch.Tensor,
        cluster_graph_edges: torch.Tensor | None,
        cluster_graph_edge_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if cluster_graph_edges is None or cluster_graph_edges.numel() == 0:
            return cluster_hidden
        edges = cluster_graph_edges.to(device=cluster_hidden.device, dtype=torch.long)
        src = edges[0].clamp(min=0, max=cluster_hidden.shape[0] - 1)
        dst = edges[1].clamp(min=0, max=cluster_hidden.shape[0] - 1)
        weights = (
            cluster_graph_edge_weights.to(device=cluster_hidden.device, dtype=cluster_hidden.dtype)
            if cluster_graph_edge_weights is not None
            else torch.ones((src.shape[0],), dtype=cluster_hidden.dtype, device=cluster_hidden.device)
        )
        messages = self.cluster_message_encoder(cluster_hidden[src]) * weights.unsqueeze(-1)
        aggregated = torch.zeros_like(cluster_hidden)
        counts = torch.zeros((cluster_hidden.shape[0], 1), dtype=cluster_hidden.dtype, device=cluster_hidden.device)
        aggregated.index_add_(0, dst, messages)
        counts.index_add_(0, dst, weights.unsqueeze(-1))
        return cluster_hidden + aggregated / counts.clamp_min(1.0)


def build_stock_graph_module(
    graph_module_type: str,
    hidden_dim: int,
    num_edge_types: int = 1,
    edge_type_prior_mode: str = "none",
) -> nn.Module:
    graph_module_type = str(graph_module_type).strip().lower()
    if graph_module_type == "mean":
        return MeanStockGraphModule(hidden_dim)
    if graph_module_type == "relation_aware":
        return RelationAwareStockGraphModule(
            hidden_dim,
            num_edge_types=num_edge_types,
            edge_type_prior_mode=edge_type_prior_mode,
        )
    if graph_module_type == "community_aware":
        return RelationAwareStockGraphModule(
            hidden_dim,
            num_edge_types=num_edge_types,
            edge_type_prior_mode=edge_type_prior_mode,
        )
    raise ValueError(
        f"Unsupported graph_module_type={graph_module_type!r}; expected one of {GRAPH_MODULE_TYPES}"
    )


class ConcatFusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_hidden: torch.Tensor,
        hyper_message: torch.Tensor,
        stock_message: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([node_hidden, hyper_message, stock_message], dim=-1)
        return self.score_head(fused).squeeze(-1)


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.modality_gate = nn.Linear(hidden_dim, 1)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_hidden: torch.Tensor,
        hyper_message: torch.Tensor,
        stock_message: torch.Tensor,
    ) -> torch.Tensor:
        modal_tokens = torch.stack([node_hidden, hyper_message, stock_message], dim=1)
        modality_weights = torch.softmax(
            self.modality_gate(modal_tokens).squeeze(-1),
            dim=1,
        ).unsqueeze(-1)
        fused = (modal_tokens * modality_weights).sum(dim=1)
        return self.score_head(fused).squeeze(-1)


class CrossEncoderFusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=_choose_attention_heads(hidden_dim),
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.modal_fusion = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.modality_gate = nn.Linear(hidden_dim, 1)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_hidden: torch.Tensor,
        hyper_message: torch.Tensor,
        stock_message: torch.Tensor,
    ) -> torch.Tensor:
        modal_tokens = torch.stack([node_hidden, hyper_message, stock_message], dim=1)
        contextual_tokens = self.modal_fusion(modal_tokens)
        modality_weights = torch.softmax(
            self.modality_gate(contextual_tokens).squeeze(-1),
            dim=1,
        ).unsqueeze(-1)
        fused = (contextual_tokens * modality_weights).sum(dim=1)
        return self.score_head(fused).squeeze(-1)


def build_fusion_module(fusion_type: str, hidden_dim: int) -> nn.Module:
    fusion_type = str(fusion_type).strip().lower()
    if fusion_type == "concat":
        return ConcatFusion(hidden_dim)
    if fusion_type == "gated":
        return GatedFusion(hidden_dim)
    if fusion_type == "cross_encoder":
        return CrossEncoderFusion(hidden_dim)
    raise ValueError(
        f"Unsupported fusion_type={fusion_type!r}; expected one of {FUSION_MODULE_TYPES}"
    )


def ranknet_pairwise_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor | None:
    if scores.numel() < 2 or targets.numel() < 2:
        return None
    score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)
    target_diff = targets.unsqueeze(0) - targets.unsqueeze(1)
    pair_mask = target_diff > 0
    if not bool(pair_mask.any()):
        return None
    return torch.nn.functional.softplus(-score_diff[pair_mask]).mean()


def _normalize_training_objective(training_objective: str) -> str:
    normalized = str(training_objective).strip().lower()
    if normalized not in TRAINING_OBJECTIVES:
        allowed = ", ".join(TRAINING_OBJECTIVES)
        raise ValueError(
            f"Unsupported training_objective={training_objective!r}; expected one of {allowed}"
        )
    return normalized


def _training_objective_loss(
    prediction: torch.Tensor,
    target_tensor: torch.Tensor,
    training_objective: str,
) -> torch.Tensor | None:
    if training_objective == "regression_mse":
        return torch.nn.functional.mse_loss(prediction, target_tensor)
    if training_objective == "ranknet_pairwise":
        return ranknet_pairwise_loss(prediction, target_tensor)
    raise AssertionError(f"Unhandled training_objective={training_objective!r}")


def build_training_label_lookup(
    labels_df: pd.DataFrame,
    training_label: str = "future_excess_return_1m",
) -> dict[tuple[str, str], float]:
    if training_label not in labels_df.columns:
        raise ValueError(
            f"Training label column {training_label!r} not found in labels_df. "
            f"Available columns: {', '.join(map(str, labels_df.columns))}"
        )
    normalized = labels_df[["month", "code", training_label]].dropna(subset=[training_label]).copy()
    normalized[training_label] = pd.to_numeric(normalized[training_label], errors="coerce")
    normalized = normalized.dropna(subset=[training_label])
    return normalized.set_index(["month", "code"])[training_label].to_dict()


def train_final_model(
    monthly_samples: dict[str, dict[str, Any]],
    labels_df: pd.DataFrame,
    train_months: list[str],
    epochs: int = 50,
    learning_rate: float = 1e-3,
    fusion_type: str = "cross_encoder",
    graph_module_type: str = "mean",
    edge_type_prior_mode: str = "none",
    community_reg_weight: float = 0.0,
    training_objective: str = "ranknet_pairwise",
    training_label: str = "future_excess_return_1m",
    seed: int = 42,
) -> DualChannelSelector:
    missing_months = sorted(set(train_months) - set(monthly_samples))
    if missing_months:
        raise ValueError(f"Training samples missing months: {missing_months}")
    if not train_months:
        raise ValueError("train_months must not be empty")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    training_objective = _normalize_training_objective(training_objective)
    label_lookup = build_training_label_lookup(labels_df, training_label)
    first_train_sample = monthly_samples[train_months[0]]
    model = DualChannelSelector(
        node_feature_dim=first_train_sample["node_features"].shape[1],
        hyperedge_feature_dim=first_train_sample["hyperedge_features"].shape[1]
        if first_train_sample["hyperedge_features"].ndim == 2
        and first_train_sample["hyperedge_features"].shape[1] > 0
        else 1,
        hidden_dim=32,
        num_industries=_infer_num_industries(monthly_samples),
        fusion_type=fusion_type,
        graph_module_type=graph_module_type,
        num_edge_types=_infer_num_edge_types(monthly_samples),
        edge_type_prior_mode=edge_type_prior_mode,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(max(epochs, 0)):
        optimizer.zero_grad()
        losses: list[torch.Tensor] = []
        for train_month in train_months:
            sample = _to_torch_sample(monthly_samples[train_month])
            scores = model(sample)
            valid_positions: list[int] = []
            targets: list[float] = []
            for index, code in enumerate(monthly_samples[train_month]["stock_codes"]):
                key = (train_month, code)
                if key in label_lookup:
                    valid_positions.append(index)
                    targets.append(label_lookup[key])
            if not valid_positions:
                continue
            loss = _training_objective_loss(
                scores[valid_positions],
                torch.tensor(targets, dtype=torch.float32),
                training_objective,
            )
            if loss is None:
                continue
            if community_reg_weight > 0:
                node_embedding = model.node_encoder(sample["node_features"])
                loss = loss + float(community_reg_weight) * community_consistency_loss(
                    node_embedding,
                    sample.get("stock_cluster_index"),
                )
            losses.append(loss)
        if not losses:
            raise ValueError(
                "No trainable labels or ranking pairs found for selected training months"
            )
        torch.stack(losses).mean().backward()
        optimizer.step()

    model.eval()
    return model


def score_month(
    model: DualChannelSelector,
    sample: dict[str, Any],
) -> pd.DataFrame:
    model.eval()
    torch_sample = _to_torch_sample(sample)
    with torch.no_grad():
        scores = model(torch_sample)
    return pd.DataFrame(
        {
            "month": [str(sample["month"])] * len(sample["stock_codes"]),
            "code": list(sample["stock_codes"]),
            "score": scores.detach().cpu().numpy().astype(float),
        }
    )


def train_walk_forward(
    monthly_samples: dict[str, dict[str, Any]],
    labels_df: pd.DataFrame,
    train_window: int = 6,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    output_path: str | Path | None = None,
    fusion_type: str = "cross_encoder",
    graph_module_type: str = "mean",
    edge_type_prior_mode: str = "none",
    community_reg_weight: float = 0.0,
    training_objective: str = "ranknet_pairwise",
    training_label: str = "future_excess_return_1m",
) -> pd.DataFrame:
    training_objective = _normalize_training_objective(training_objective)
    months = sorted(monthly_samples)
    if output_path is not None:
        _write_frame(pd.DataFrame(columns=["month", "code", "score"]), Path(output_path))
    if len(months) < 2:
        return pd.DataFrame(columns=["month", "code", "score"])

    label_lookup = build_training_label_lookup(labels_df, training_label)
    num_industries = _infer_num_industries(monthly_samples)
    num_edge_types = _infer_num_edge_types(monthly_samples)
    score_rows: list[dict[str, Any]] = []

    for test_index, test_month in enumerate(months):
        train_months = months[:test_index]
        if train_window > 0:
            train_months = train_months[-train_window:]
        if not train_months:
            continue

        first_train_sample = monthly_samples[train_months[0]]
        model = DualChannelSelector(
            node_feature_dim=first_train_sample["node_features"].shape[1],
            hyperedge_feature_dim=first_train_sample["hyperedge_features"].shape[1]
            if first_train_sample["hyperedge_features"].ndim == 2
            and first_train_sample["hyperedge_features"].shape[1] > 0
            else 1,
            hidden_dim=32,
            num_industries=num_industries,
            fusion_type=fusion_type,
            graph_module_type=graph_module_type,
            num_edge_types=num_edge_types,
            edge_type_prior_mode=edge_type_prior_mode,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for _ in range(max(epochs, 0)):
            optimizer.zero_grad()
            losses: list[torch.Tensor] = []
            for train_month in train_months:
                sample = _to_torch_sample(monthly_samples[train_month])
                scores = model(sample)
                targets = []
                valid_positions = []
                for index, code in enumerate(monthly_samples[train_month]["stock_codes"]):
                    key = (train_month, code)
                    if key in label_lookup:
                        valid_positions.append(index)
                        targets.append(label_lookup[key])
                if not valid_positions:
                    continue
                prediction = scores[valid_positions]
                target_tensor = torch.tensor(targets, dtype=torch.float32)
                loss = _training_objective_loss(
                    prediction,
                    target_tensor,
                    training_objective,
                )
                if loss is None:
                    continue
                if community_reg_weight > 0:
                    node_embedding = model.node_encoder(sample["node_features"])
                    loss = loss + float(community_reg_weight) * community_consistency_loss(
                        node_embedding,
                        sample.get("stock_cluster_index"),
                    )
                losses.append(loss)
            if not losses:
                break
            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            optimizer.step()

        sample = _to_torch_sample(monthly_samples[test_month])
        with torch.no_grad():
            scores = model(sample)
        for code, score in zip(monthly_samples[test_month]["stock_codes"], scores.cpu().numpy(), strict=True):
            score_rows.append({"month": test_month, "code": code, "score": float(score)})

    scores_df = pd.DataFrame(score_rows)
    if output_path is not None:
        _write_frame(scores_df, Path(output_path))
    return scores_df


def _month_distance(start_month: str, end_month: str) -> int:
    start_year, start_mon = map(int, start_month.split("-"))
    end_year, end_mon = map(int, end_month.split("-"))
    return (end_year - start_year) * 12 + (end_mon - start_mon)


def _build_cluster_sample_arrays(
    month: str,
    stock_codes: list[str],
    month_clusters: pd.DataFrame,
    cluster_edges_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if month_clusters.empty or "cluster_id" not in month_clusters.columns:
        return (
            np.zeros((len(stock_codes),), dtype=np.int64),
            np.zeros((0, 5), dtype=np.float32),
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    clusters = month_clusters.copy()
    clusters["code"] = clusters["code"].map(_normalize_stock_code)
    clusters["cluster_id"] = pd.to_numeric(clusters["cluster_id"], errors="coerce").fillna(0).astype(int)
    unique_cluster_ids = sorted(clusters["cluster_id"].unique().tolist())
    cluster_to_index = {cluster_id: index for index, cluster_id in enumerate(unique_cluster_ids)}
    code_to_cluster = clusters.set_index("code")["cluster_id"].to_dict()
    stock_cluster_index = np.array(
        [cluster_to_index.get(int(code_to_cluster.get(code, unique_cluster_ids[0])), 0) for code in stock_codes],
        dtype=np.int64,
    )
    cluster_rows = []
    for cluster_id in unique_cluster_ids:
        row = clusters.loc[clusters["cluster_id"] == cluster_id].iloc[0]
        cluster_rows.append(
            [
                float(pd.to_numeric(row.get("cluster_size", 0.0), errors="coerce") or 0.0),
                float(pd.to_numeric(row.get("cluster_score", 0.0), errors="coerce") or 0.0),
                float(pd.to_numeric(row.get("pagerank", 0.0), errors="coerce") or 0.0),
                float(pd.to_numeric(row.get("weighted_degree", 0.0), errors="coerce") or 0.0),
                float(pd.to_numeric(row.get("report_attention", 0.0), errors="coerce") or 0.0),
            ]
        )
    cluster_features = np.array(cluster_rows, dtype=np.float32)

    edge_frame = cluster_edges_df.loc[
        cluster_edges_df.get("month", pd.Series(dtype=str)).astype(str).eq(str(month))
    ].copy() if not cluster_edges_df.empty else pd.DataFrame()
    edge_pairs: list[list[int]] = [[], []]
    edge_weights: list[float] = []
    if not edge_frame.empty:
        for row in edge_frame.itertuples(index=False):
            src_raw = int(getattr(row, "src_cluster_id"))
            dst_raw = int(getattr(row, "dst_cluster_id"))
            if src_raw not in cluster_to_index or dst_raw not in cluster_to_index:
                continue
            src = cluster_to_index[src_raw]
            dst = cluster_to_index[dst_raw]
            weight = float(getattr(row, "edge_weight", 1.0) or 1.0)
            edge_pairs[0].extend([src, dst])
            edge_pairs[1].extend([dst, src])
            edge_weights.extend([weight, weight])
    cluster_graph_edges = np.array(edge_pairs, dtype=np.int64)
    if cluster_graph_edges.size == 0:
        cluster_graph_edges = np.zeros((2, 0), dtype=np.int64)
    return (
        stock_cluster_index,
        cluster_features,
        cluster_graph_edges,
        np.array(edge_weights, dtype=np.float32),
    )


def community_consistency_loss(
    node_embeddings: torch.Tensor,
    stock_cluster_index: torch.Tensor | None,
) -> torch.Tensor:
    if (
        stock_cluster_index is None
        or node_embeddings.numel() == 0
        or stock_cluster_index.numel() == 0
        or node_embeddings.shape[0] < 2
    ):
        return node_embeddings.new_tensor(0.0)
    cluster_index = stock_cluster_index.to(device=node_embeddings.device, dtype=torch.long)
    unique_clusters = torch.unique(cluster_index)
    losses: list[torch.Tensor] = []
    centers: list[torch.Tensor] = []
    for cluster_id in unique_clusters:
        mask = cluster_index == cluster_id
        members = node_embeddings[mask]
        if members.numel() == 0:
            continue
        center = members.mean(dim=0)
        centers.append(center)
        if members.shape[0] > 1:
            losses.append(((members - center) ** 2).mean())
    if len(centers) > 1:
        center_tensor = torch.stack(centers)
        distances = torch.cdist(center_tensor, center_tensor, p=2)
        off_diagonal = distances[~torch.eye(len(centers), dtype=torch.bool, device=distances.device)]
        losses.append(torch.relu(1.0 - off_diagonal).mean())
    if not losses:
        return node_embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()


def _is_active_report_month(report_month: Any, target_month: str, decay_months: int) -> bool:
    if pd.isna(report_month):
        return False
    raw = str(report_month).strip()
    if not raw:
        return False
    try:
        distance = _month_distance(raw, target_month)
    except Exception:
        return False
    return 0 <= distance < decay_months


def _available_month_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=str)
    fallback = (
        frame["report_month"]
        if "report_month" in frame.columns
        else pd.Series([""] * len(frame), index=frame.index)
    )
    fallback = fallback.map(_normalize_month_value)
    if "available_month" in frame.columns:
        available = frame["available_month"].map(_normalize_month_value)
        fallback = available.where(available.astype(bool), fallback)
    if "available_date" in frame.columns:
        available_from_date = frame["available_date"].map(_month_from_date_value)
        fallback = available_from_date.where(available_from_date.astype(bool), fallback)
    return fallback.astype(str)


def _normalize_month_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    try:
        return pd.Period(raw, freq="M").strftime("%Y-%m")
    except Exception:
        return raw


def _month_from_date_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Period(parsed, freq="M").strftime("%Y-%m")


def _is_rolling_report_edge_type(edge_types: pd.Series) -> pd.Series:
    if edge_types.empty:
        return pd.Series(dtype=bool)
    return edge_types.fillna("").astype(str).isin({"same_industry", "text_similarity"})


def _ensure_non_empty_feature_matrix(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 2 and values.shape[0] >= 0 and values.shape[1] > 0:
        return values
    if values.ndim == 2 and values.shape[0] == 0:
        return torch.zeros((0, 1), dtype=values.dtype, device=values.device)
    return values.reshape(values.shape[0], -1)


def _choose_attention_heads(hidden_dim: int) -> int:
    for num_heads in (8, 4, 2):
        if hidden_dim % num_heads == 0:
            return num_heads
    return 1


def _edge_softmax(
    attention_logits: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    attention_weights = torch.zeros_like(attention_logits)
    for node_index in range(num_nodes):
        mask = dst == node_index
        if torch.any(mask):
            attention_weights[mask] = torch.softmax(attention_logits[mask], dim=0)
    return attention_weights


def _normalize_edge_type_ids(
    edge_type_ids: torch.Tensor | None,
    num_edges: int,
    max_edge_type_id: int,
    device: torch.device,
) -> torch.Tensor:
    if edge_type_ids is None:
        return torch.zeros((num_edges,), dtype=torch.long, device=device)
    return edge_type_ids.to(device=device, dtype=torch.long).clamp(min=0, max=max_edge_type_id)


def _normalize_edge_weights(
    edge_weights: torch.Tensor | None,
    num_edges: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if edge_weights is None:
        return torch.ones((num_edges,), dtype=dtype, device=device)
    return edge_weights.to(device=device, dtype=dtype)


def _to_torch_sample(sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "node_features": torch.tensor(sample["node_features"], dtype=torch.float32),
        "industry_id": torch.tensor(
            sample.get("industry_id", np.zeros(sample["node_features"].shape[0], dtype=np.int64)),
            dtype=torch.long,
        ),
        "hyperedge_features": torch.tensor(sample["hyperedge_features"], dtype=torch.float32),
        "incidence_index": torch.tensor(sample["incidence_index"], dtype=torch.long),
        "incidence_weight": torch.tensor(sample["incidence_weight"], dtype=torch.float32),
        "stock_graph_edges": torch.tensor(sample["stock_graph_edges"], dtype=torch.long),
        "stock_graph_edge_type_ids": torch.tensor(
            sample.get(
                "stock_graph_edge_type_ids",
                np.zeros(sample["stock_graph_edges"].shape[1], dtype=np.int64),
            ),
            dtype=torch.long,
        ),
        "stock_graph_edge_weights": torch.tensor(
            sample.get(
                "stock_graph_edge_weights",
                np.ones(sample["stock_graph_edges"].shape[1], dtype=np.float32),
            ),
            dtype=torch.float32,
        ),
        "stock_cluster_index": torch.tensor(
            sample.get(
                "stock_cluster_index",
                np.zeros(sample["node_features"].shape[0], dtype=np.int64),
            ),
            dtype=torch.long,
        ),
        "cluster_features": torch.tensor(
            sample.get("cluster_features", np.zeros((0, 5), dtype=np.float32)),
            dtype=torch.float32,
        ),
        "cluster_graph_edges": torch.tensor(
            sample.get("cluster_graph_edges", np.zeros((2, 0), dtype=np.int64)),
            dtype=torch.long,
        ),
        "cluster_graph_edge_weights": torch.tensor(
            sample.get("cluster_graph_edge_weights", np.zeros((0,), dtype=np.float32)),
            dtype=torch.float32,
        ),
    }


def _infer_num_industries(monthly_samples: dict[str, dict[str, Any]]) -> int:
    max_industry_id = 0
    for sample in monthly_samples.values():
        industry_id = sample.get("industry_id")
        if industry_id is None or len(industry_id) == 0:
            continue
        max_industry_id = max(max_industry_id, int(np.max(industry_id)))
    return max_industry_id + 1


def _infer_num_edge_types(monthly_samples: dict[str, dict[str, Any]]) -> int:
    max_edge_type_id = 0
    for sample in monthly_samples.values():
        edge_type_ids = sample.get("stock_graph_edge_type_ids")
        if edge_type_ids is None or len(edge_type_ids) == 0:
            continue
        max_edge_type_id = max(max_edge_type_id, int(np.max(edge_type_ids)))
    return max_edge_type_id + 1


def _parse_stock_edge_builders(edge_builders: str | list[str]) -> list[str]:
    if isinstance(edge_builders, str):
        raw_builders = [part.strip() for part in edge_builders.split(",")]
    else:
        raw_builders = [str(part).strip() for part in edge_builders]
    builders = [builder for builder in raw_builders if builder]
    unsupported = sorted(set(builders) - set(STOCK_EDGE_BUILDERS))
    if unsupported:
        allowed = ", ".join(STOCK_EDGE_BUILDERS)
        raise ValueError(f"Unsupported stock edge builders: {unsupported}; allowed builders: {allowed}")
    return builders


def _empty_stock_edges() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["month", "src_code", "dst_code", "edge_type", "edge_weight", "source_id", "report_id"]
    )


def _normalize_stock_edge_frame(edges_df: pd.DataFrame) -> pd.DataFrame:
    if edges_df.empty:
        return _empty_stock_edges()
    normalized = edges_df.copy()
    if "month" not in normalized.columns:
        normalized["month"] = ""
    if "report_id" not in normalized.columns:
        normalized["report_id"] = ""
    if "source_id" not in normalized.columns:
        normalized["source_id"] = normalized["report_id"]
    normalized["source_id"] = normalized["source_id"].fillna("").astype(str)
    normalized["report_id"] = normalized["report_id"].fillna("").astype(str)
    normalized["month"] = normalized["month"].fillna("").astype(str)
    normalized["src_code"] = normalized["src_code"].map(_normalize_stock_code)
    normalized["dst_code"] = normalized["dst_code"].map(_normalize_stock_code)
    normalized["edge_weight"] = pd.to_numeric(normalized.get("edge_weight", 1.0), errors="coerce").fillna(1.0)
    return normalized.reindex(columns=_empty_stock_edges().columns)


def _kg_time_decay(report_month: str, target_month: str | None, kg_decay_months: int) -> float:
    if not target_month:
        return 1.0
    try:
        distance = _month_distance(str(report_month), str(target_month))
    except Exception:
        return 0.0
    if distance < 0:
        return 0.0
    decay_months = max(int(kg_decay_months), 1)
    return float(np.exp(-distance / decay_months))


def _llm_stock_pair_edges(
    relations: pd.DataFrame,
    stock_codes: set[str],
    month: str,
    decay_months: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edge_type_by_relation = {
        "core_subject": "kg_llm_core_subject",
        "peer": "kg_llm_peer",
        "positive_driver": "kg_llm_positive_driver",
        "risk_pressure": "kg_llm_risk_pressure",
    }
    pair_relations = relations.loc[relations["relation_type"].isin(edge_type_by_relation)].copy()
    for relation in pair_relations.to_dict(orient="records"):
        src_code = _normalize_stock_code(relation.get("head_code", ""))
        dst_code = _normalize_stock_code(relation.get("tail_code", ""))
        if not src_code or not dst_code or src_code == dst_code:
            continue
        if src_code not in stock_codes or dst_code not in stock_codes:
            continue
        edge_type = edge_type_by_relation[str(relation.get("relation_type", ""))]
        edge_weight = _llm_relation_weight(relation, month, decay_months)
        for left, right in ((src_code, dst_code), (dst_code, src_code)):
            rows.append(
                {
                    "month": month,
                    "src_code": left,
                    "dst_code": right,
                    "edge_type": edge_type,
                    "edge_weight": edge_weight,
                    "source_id": _llm_relation_source_id(relation),
                    "report_id": str(relation.get("report_id", "")),
                }
            )
    return rows


def _llm_relation_source_id(relation: dict[str, Any]) -> str:
    parts = [
        "kg_llm",
        str(relation.get("report_id", "")),
        str(relation.get("relation_type", "")),
        str(relation.get("topic", "")),
        str(relation.get("driver_type", "")),
        str(relation.get("risk_type", "")),
    ]
    return ":".join(part for part in parts if part)


def _llm_relation_weight(relation: dict[str, Any], target_month: str, decay_months: int) -> float:
    confidence = _safe_float(relation.get("confidence", 0.0), default=0.0)
    evidence_score = _llm_evidence_score(str(relation.get("source_section", "")))
    availability_month = str(
        relation.get("available_month", relation.get("report_month", ""))
    )
    decay = _kg_time_decay(availability_month, target_month, decay_months)
    return max(float(confidence * evidence_score * decay), 1e-6)


def _llm_evidence_score(source_section: str) -> float:
    normalized = source_section.strip().lower()
    if normalized in {"title", "conclusion", "summary"}:
        return 1.0
    if normalized in {"body", "raw_text", "text"}:
        return 0.8
    return 0.5


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    if np.isnan(parsed):
        return default
    return parsed


def _build_topk_similarity_edges(
    monthly_features_df: pd.DataFrame,
    daily_k_df: pd.DataFrame,
    hs300_daily_df: pd.DataFrame | None,
    lookback_days: int,
    topk: int,
    min_score: float,
    edge_type: str,
    same_industry_only: bool,
) -> pd.DataFrame:
    if monthly_features_df.empty or daily_k_df.empty or topk <= 0:
        return _empty_stock_edges()

    monthly = monthly_features_df.copy()
    monthly["code"] = monthly["code"].map(_normalize_stock_code)
    monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce")
    daily = _prepare_market_daily_features(daily_k_df, hs300_daily_df)
    rows: list[dict[str, Any]] = []

    for month, month_rows in monthly.groupby("month"):
        month_rows = month_rows.dropna(subset=["date"]).sort_values("code")
        codes = [code for code in month_rows["code"].tolist() if code]
        if len(codes) < 2:
            continue
        month_end = month_rows["date"].max()
        window = daily.loc[
            (daily["code"].isin(codes))
            & (daily["date"] <= month_end)
        ].copy()
        if window.empty:
            continue

        industry_lookup = (
            month_rows.set_index("code").get("industry", pd.Series(dtype=str)).fillna("").astype(str).to_dict()
            if "industry" in month_rows.columns
            else {}
        )
        vectors = {
            code: _stock_market_vector(frame.tail(max(int(lookback_days), 1)))
            for code, frame in window.sort_values("date").groupby("code")
        }
        for src_code in codes:
            src_vector = vectors.get(src_code)
            if src_vector is None:
                continue
            candidates: list[tuple[str, float]] = []
            for dst_code in codes:
                if dst_code == src_code:
                    continue
                if same_industry_only and (
                    not industry_lookup.get(src_code)
                    or industry_lookup.get(src_code) != industry_lookup.get(dst_code)
                ):
                    continue
                dst_vector = vectors.get(dst_code)
                if dst_vector is None:
                    continue
                score = _safe_corr(src_vector, dst_vector)
                if score > float(min_score):
                    candidates.append((dst_code, score))
            for dst_code, score in sorted(candidates, key=lambda item: item[1], reverse=True)[: int(topk)]:
                rows.append(
                    {
                        "month": month,
                        "src_code": src_code,
                        "dst_code": dst_code,
                        "edge_type": edge_type,
                        "edge_weight": float(score),
                        "source_id": f"{edge_type}:{month}",
                        "report_id": "",
                    }
                )
    if not rows:
        return _empty_stock_edges()
    return pd.DataFrame(rows, columns=_empty_stock_edges().columns)


def _prepare_market_daily_features(
    daily_k_df: pd.DataFrame,
    hs300_daily_df: pd.DataFrame | None,
) -> pd.DataFrame:
    daily = daily_k_df.copy()
    if daily.empty:
        return pd.DataFrame()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["code"] = daily["code"].map(_normalize_stock_code)
    for column in ["close", "high", "low", "amount", "turn", "volume"]:
        if column not in daily.columns:
            daily[column] = 0.0
        daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0.0)
    daily = daily.sort_values(["code", "date"], ignore_index=True)
    daily["daily_return"] = daily.groupby("code")["close"].pct_change().fillna(0.0)
    daily["abs_return"] = daily["daily_return"].abs()
    daily["intraday_range"] = (
        (daily["high"] - daily["low"]) / daily["close"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily["log_amount"] = np.log1p(daily["amount"].clip(lower=0.0))
    daily["turn"] = daily["turn"].fillna(0.0)

    benchmark = _benchmark_returns(hs300_daily_df)
    if not benchmark.empty:
        daily = daily.merge(benchmark, on="date", how="left")
    else:
        daily["benchmark_return"] = 0.0
    daily["benchmark_return"] = daily["benchmark_return"].fillna(0.0)
    daily["excess_return_hs300"] = daily["daily_return"] - daily["benchmark_return"]
    return daily


def _benchmark_returns(hs300_daily_df: pd.DataFrame | None) -> pd.DataFrame:
    if hs300_daily_df is None or hs300_daily_df.empty:
        return pd.DataFrame(columns=["date", "benchmark_return"])
    benchmark = hs300_daily_df.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark["close"] = pd.to_numeric(benchmark.get("close"), errors="coerce")
    benchmark = benchmark.dropna(subset=["date", "close"]).sort_values("date")
    benchmark["benchmark_return"] = benchmark["close"].pct_change().fillna(0.0)
    return benchmark[["date", "benchmark_return"]].drop_duplicates(subset=["date"])


def _stock_market_vector(frame: pd.DataFrame) -> np.ndarray:
    columns = [
        "daily_return",
        "excess_return_hs300",
        "abs_return",
        "intraday_range",
        "turn",
        "log_amount",
    ]
    values = frame.reindex(columns=columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return values.to_numpy(dtype=np.float32).reshape(-1)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    size = min(left.size, right.size)
    if size < 2:
        return 0.0
    left = left[-size:]
    right = right[-size:]
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    score = float(np.corrcoef(left, right)[0, 1])
    if np.isnan(score):
        return 0.0
    return score


def _normalize_stock_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    return normalize_baostock_code(raw)


def _write_frame(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
    except Exception:
        df.to_pickle(output_path)
