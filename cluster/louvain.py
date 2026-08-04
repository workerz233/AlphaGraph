from __future__ import annotations

from typing import Any

import networkx as nx
from networkx.algorithms.community import louvain_communities


class LouvainClusterer:
    name = "louvain"
    supports_weighted_graph = True

    def default_params(self) -> dict[str, Any]:
        return {"resolution": 1.2, "seed": 42}

    def fit_predict(
        self,
        graph: nx.Graph,
        node_features: dict[str, dict[str, Any]],
        config: dict[str, Any],
    ) -> dict[str, int]:
        del node_features
        if graph.number_of_nodes() == 0:
            return {}
        params = self.default_params()
        params.update(config)
        communities = louvain_communities(
            graph,
            weight="weight",
            resolution=float(params.get("resolution", 1.2)),
            seed=int(params.get("seed", 42)),
        )
        return {
            str(code): cluster_id
            for cluster_id, community in enumerate(
                sorted(communities, key=lambda nodes: sorted(str(node) for node in nodes)[0])
            )
            for code in community
        }
