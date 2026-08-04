from __future__ import annotations

from typing import Any

import networkx as nx


class LeidenClusterer:
    name = "leiden"
    supports_weighted_graph = True

    def default_params(self) -> dict[str, Any]:
        return {"resolution": 1.2, "seed": 42, "n_iterations": 2}

    def fit_predict(
        self,
        graph: nx.Graph,
        node_features: dict[str, dict[str, Any]],
        config: dict[str, Any],
    ) -> dict[str, int]:
        del node_features
        if graph.number_of_nodes() == 0:
            return {}
        if graph.number_of_edges() == 0:
            return {str(code): index for index, code in enumerate(sorted(str(node) for node in graph.nodes))}

        try:
            import igraph as ig
            import leidenalg
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised when optional deps are absent
            raise RuntimeError(
                "cluster_method='leiden' requires optional dependencies: "
                "pip install igraph leidenalg"
            ) from exc

        params = self.default_params()
        params.update(config)
        nodes = sorted(str(node) for node in graph.nodes)
        node_to_index = {node: index for index, node in enumerate(nodes)}
        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        for src, dst, data in graph.edges(data=True):
            edges.append((node_to_index[str(src)], node_to_index[str(dst)]))
            weights.append(float(data.get("weight", 1.0)))

        ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
        ig_graph.vs["name"] = nodes
        ig_graph.es["weight"] = weights
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(params.get("resolution", 1.2)),
            seed=int(params.get("seed", 42)),
            n_iterations=int(params.get("n_iterations", 2)),
        )
        communities = [
            [str(ig_graph.vs[index]["name"]) for index in community]
            for community in partition
        ]
        return {
            code: cluster_id
            for cluster_id, community in enumerate(
                sorted(communities, key=lambda names: sorted(names)[0])
            )
            for code in community
        }
