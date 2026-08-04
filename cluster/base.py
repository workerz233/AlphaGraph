from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import networkx as nx


@dataclass(frozen=True)
class ClusterResult:
    labels: dict[str, int]
    params: dict[str, Any]


class Clusterer(Protocol):
    name: str
    supports_weighted_graph: bool

    def fit_predict(
        self,
        graph: nx.Graph,
        node_features: dict[str, dict[str, Any]],
        config: dict[str, Any],
    ) -> dict[str, int]:
        ...

    def default_params(self) -> dict[str, Any]:
        ...
