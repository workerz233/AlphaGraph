from __future__ import annotations

import json

import pandas as pd

from alphagraph.kg import build as kg_build
from alphagraph.kg.build import build_kg_artifacts
from alphagraph.kg.extractor import LLM_KG_COLUMNS, extract_report_llm_relations


def test_extract_report_llm_relations_emits_per_report_progress(monkeypatch):
    reports = pd.DataFrame(
        [
            {
                "report_id": "r1",
                "report_month": "202501",
                "raw_text": "## 关联标的\n平安银行（000001.SZ）\n浦发银行（600000.SH）\n共同受益于信贷修复。",
            }
        ]
    )
    response = {
        "relations": [
            {
                "relation_type": "positive_driver",
                "head_type": "stock",
                "head_code": "000001.SZ",
                "head_name": "平安银行",
                "tail_type": "stock",
                "tail_code": "600000.SH",
                "tail_name": "浦发银行",
                "topic": "信贷修复",
                "driver_type": "macro",
                "risk_type": "",
                "sentiment": "positive",
                "confidence": 0.9,
                "evidence_text": "共同受益于信贷修复",
                "source_section": "summary",
            }
        ]
    }

    monkeypatch.setattr(
        "alphagraph.kg.extractor.call_openai_compatible_chat",
        lambda **kwargs: json.dumps(response, ensure_ascii=False),
    )
    progress: list[str] = []

    relations = extract_report_llm_relations(
        reports_df=reports,
        base_url="https://example.test",
        api_key="test-key",
        model="test-model",
        progress_callback=progress.append,
    )

    assert len(relations) == 1
    assert any("1/1 请求API：report_id=r1" in message for message in progress)
    assert any("1/1 解析完成：report_id=r1 relations=1" in message for message in progress)


def test_build_kg_artifacts_keeps_legacy_extractor_fn_compatible(tmp_path, monkeypatch):
    reports = pd.DataFrame([{"report_id": "r1", "report_month": "2025-01", "raw_text": "text"}])
    monthly_features = pd.DataFrame(
        [
            {"month": "2025-01", "date": "2025-01-31", "code": "sz.000001"},
            {"month": "2025-01", "date": "2025-01-31", "code": "sh.600000"},
        ]
    )
    relation = pd.DataFrame(
        [
            {
                "report_id": "r1",
                "report_month": "2025-01",
                "relation_type": "peer",
                "head_type": "stock",
                "head_code": "sz.000001",
                "head_name": "平安银行",
                "tail_type": "stock",
                "tail_code": "sh.600000",
                "tail_name": "浦发银行",
                "topic": "同业",
                "driver_type": "",
                "risk_type": "",
                "sentiment": "neutral",
                "confidence": 0.8,
                "evidence_text": "",
                "source_section": "",
            }
        ],
        columns=LLM_KG_COLUMNS,
    )

    input_paths = {
        "reports": tmp_path / "reports.parquet",
        "incidence": tmp_path / "incidence.parquet",
        "monthly_features": tmp_path / "monthly_features.parquet",
        "daily_k": tmp_path / "daily_k.parquet",
        "hs300_daily_k": tmp_path / "hs300_daily_k.parquet",
    }
    reports.to_pickle(input_paths["reports"])
    pd.DataFrame().to_pickle(input_paths["incidence"])
    monthly_features.to_pickle(input_paths["monthly_features"])
    pd.DataFrame().to_pickle(input_paths["daily_k"])
    pd.DataFrame().to_pickle(input_paths["hs300_daily_k"])

    def legacy_extractor_fn(reports_df, base_url, api_key, model):
        assert len(reports_df) == 1
        assert base_url == "https://example.test"
        assert api_key == "test-key"
        assert model == "test-model"
        return relation

    monkeypatch.setattr("alphagraph.kg.build.write_kg_html_report", lambda **kwargs: None)
    progress: list[str] = []

    result = build_kg_artifacts(
        reports_path=input_paths["reports"],
        incidence_path=input_paths["incidence"],
        monthly_features_path=input_paths["monthly_features"],
        daily_k_path=input_paths["daily_k"],
        hs300_daily_k_path=input_paths["hs300_daily_k"],
        report_embeddings_path=tmp_path / "report_embeddings.parquet",
        report_llm_relations_path=tmp_path / "report_llm_relations.parquet",
        kg_stock_edges_llm_path=tmp_path / "kg_stock_edges_llm.parquet",
        stock_graph_edges_path=tmp_path / "stock_graph_edges.parquet",
        kg_html_path=tmp_path / "kg_graph.html",
        stock_edge_builders="kg_llm",
        llm_kg_base_url="https://example.test",
        llm_kg_api_key="test-key",
        llm_kg_model="test-model",
        embedding_fn=lambda texts, model_name: [[0.1, 0.2] for _ in texts],
        extractor_fn=legacy_extractor_fn,
        progress_fn=progress.append,
    )

    assert result["relation_rows"] == 1
    assert result["kg_edge_rows"] == 2
    assert any("开始 LLM 关系抽取" in message for message in progress)


def test_build_kg_artifacts_refreshes_report_caches_when_report_ids_change(
    tmp_path,
    monkeypatch,
):
    reports = pd.DataFrame(
        [
            {"report_id": "r1", "report_month": "2025-03", "raw_text": "first"},
            {"report_id": "r2", "report_month": "2025-04", "raw_text": "second"},
        ]
    )
    monthly_features = pd.DataFrame(
        [{"month": "2025-04", "date": "2025-04-30", "code": "sz.000001"}]
    )
    input_paths = {
        "reports": tmp_path / "reports.parquet",
        "incidence": tmp_path / "incidence.parquet",
        "monthly_features": tmp_path / "monthly_features.parquet",
        "daily_k": tmp_path / "daily_k.parquet",
        "hs300_daily_k": tmp_path / "hs300_daily_k.parquet",
    }
    reports.to_pickle(input_paths["reports"])
    pd.DataFrame().to_pickle(input_paths["incidence"])
    monthly_features.to_pickle(input_paths["monthly_features"])
    pd.DataFrame().to_pickle(input_paths["daily_k"])
    pd.DataFrame().to_pickle(input_paths["hs300_daily_k"])

    embeddings_path = tmp_path / "report_embeddings.parquet"
    relations_path = tmp_path / "report_llm_relations.parquet"
    pd.DataFrame([{"report_id": "r1", "emb_0": 0.1}]).to_pickle(embeddings_path)
    pd.DataFrame(columns=LLM_KG_COLUMNS).to_pickle(relations_path)
    embedded_texts: list[str] = []
    extracted_report_ids: list[str] = []

    def embedding_fn(texts, model_name):
        del model_name
        embedded_texts.extend(texts)
        return [[0.1, 0.2] for _ in texts]

    def extractor_fn(reports_df, **kwargs):
        del kwargs
        extracted_report_ids.extend(reports_df["report_id"].tolist())
        return pd.DataFrame(columns=LLM_KG_COLUMNS)

    monkeypatch.setattr("alphagraph.kg.build.write_kg_html_report", lambda **kwargs: None)
    progress: list[str] = []

    result = build_kg_artifacts(
        reports_path=input_paths["reports"],
        incidence_path=input_paths["incidence"],
        monthly_features_path=input_paths["monthly_features"],
        daily_k_path=input_paths["daily_k"],
        hs300_daily_k_path=input_paths["hs300_daily_k"],
        report_embeddings_path=embeddings_path,
        report_llm_relations_path=relations_path,
        kg_stock_edges_llm_path=tmp_path / "kg_stock_edges_llm.parquet",
        stock_graph_edges_path=tmp_path / "stock_graph_edges.parquet",
        kg_html_path=tmp_path / "kg_graph.html",
        stock_edge_builders="kg_llm",
        embedding_fn=embedding_fn,
        extractor_fn=extractor_fn,
        progress_fn=progress.append,
    )

    assert result["embedding_rows"] == 2
    assert embedded_texts == ["first", "second"]
    assert extracted_report_ids == ["r1", "r2"]
    assert any("报告集合已变化" in message for message in progress)


def test_embedding_cache_requires_matching_model_and_report_text():
    reports = pd.DataFrame([{"report_id": "r1", "raw_text": "original"}])
    embeddings = pd.DataFrame([{"report_id": "r1", "emb_0": 0.1}])
    cached = kg_build._add_embedding_cache_metadata(
        embeddings,
        reports,
        model_name="model-a",
    )

    assert kg_build._embedding_cache_matches(reports, cached, model_name="model-a")
    assert not kg_build._embedding_cache_matches(reports, cached, model_name="model-b")
    changed = reports.assign(raw_text="changed")
    assert not kg_build._embedding_cache_matches(changed, cached, model_name="model-a")
