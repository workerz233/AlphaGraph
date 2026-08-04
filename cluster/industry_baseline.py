from __future__ import annotations

from typing import Any

import networkx as nx


class IndustryBaselineClusterer:
    name = "industry_baseline"
    supports_weighted_graph = False

    def default_params(self) -> dict[str, Any]:
        return {}

    def fit_predict(
        self,
        graph: nx.Graph,
        node_features: dict[str, dict[str, Any]],
        config: dict[str, Any],
    ) -> dict[str, int]:
        del config
        industry_to_id: dict[str, int] = {}
        labels: dict[str, int] = {}
        for code in sorted(str(node) for node in graph.nodes):
            industry = str(node_features.get(code, {}).get("industry", "")).strip() or "unknown"
            if industry not in industry_to_id:
                industry_to_id[industry] = len(industry_to_id)
            labels[code] = industry_to_id[industry]
        return labels
