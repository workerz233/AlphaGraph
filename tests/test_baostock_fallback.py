from __future__ import annotations

from alphagraph.graph.build import _should_use_tushare_fallback


def test_tushare_fallback_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HYPERGRAPH_ENABLE_TUSHARE_FALLBACK", raising=False)

    error = RuntimeError(
        "query_history_k_data_plus failed for sh.600008: "
        "10001001 you don't login."
    )

    assert _should_use_tushare_fallback(error) is False
