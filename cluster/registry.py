from __future__ import annotations

from alphagraph.cluster.industry_baseline import IndustryBaselineClusterer
from alphagraph.cluster.label_propagation import LabelPropagationClusterer
from alphagraph.cluster.leiden import LeidenClusterer
from alphagraph.cluster.louvain import LouvainClusterer


_CLUSTERERS = {
    clusterer.name: clusterer
    for clusterer in (
        LeidenClusterer(),
        LouvainClusterer(),
        LabelPropagationClusterer(),
        IndustryBaselineClusterer(),
    )
}


def available_clusterers() -> list[str]:
    return sorted(_CLUSTERERS)


def get_clusterer(name: str):
    method = str(name).strip().lower()
    if method not in _CLUSTERERS:
        allowed = ", ".join(available_clusterers())
        raise ValueError(f"Unsupported cluster_method={name!r}; allowed methods: {allowed}")
    return _CLUSTERERS[method]
