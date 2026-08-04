from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from alphagraph.gnn.predict import (
    build_checkpoint_metadata,
    checkpoint_path_for_training_end,
    load_prediction_checkpoint,
    save_prediction_checkpoint,
    select_training_months,
)
from alphagraph.model.core import (
    DualChannelSelector,
    build_monthly_samples,
    score_month,
    train_final_model,
)


def _sample(month: str) -> dict[str, object]:
    return {
        "month": month,
        "stock_codes": ["sz.000001", "sz.000002"],
        "node_features": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "industry_id": np.array([0, 1], dtype=np.int64),
        "hyperedge_features": np.zeros((0, 3), dtype=np.float32),
        "incidence_index": np.zeros((2, 0), dtype=np.int64),
        "incidence_weight": np.zeros((0,), dtype=np.float32),
        "stock_graph_edges": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "stock_graph_edge_type_ids": np.array([0, 0], dtype=np.int64),
        "stock_graph_edge_weights": np.ones((2,), dtype=np.float32),
        "stock_graph_edge_type_to_id": {"market_corr_topk": 0},
        "stock_cluster_index": np.zeros((0,), dtype=np.int64),
        "cluster_features": np.zeros((0, 5), dtype=np.float32),
        "cluster_graph_edges": np.zeros((2, 0), dtype=np.int64),
        "cluster_graph_edge_weights": np.zeros((0,), dtype=np.float32),
    }


def test_select_training_months_uses_explicit_range_and_excludes_signal_month():
    samples = {month: _sample(month) for month in ("2026-05", "2026-06", "2026-07")}
    labels = pd.DataFrame(
        [
            {"month": "2026-05", "code": "sz.000001", "future_excess_return_1m": 0.1},
            {"month": "2026-06", "code": "sz.000001", "future_excess_return_1m": 0.2},
            {"month": "2026-07", "code": "sz.000001", "future_excess_return_1m": 0.3},
        ]
    )

    months = select_training_months(
        samples,
        labels,
        signal_month="202607",
        train_start_month="202605",
        train_end_month="202606",
        training_label="future_excess_return_1m",
    )

    assert months == ["2026-05", "2026-06"]


def test_checkpoint_path_uses_training_quarter_and_as_of_date(tmp_path: Path):
    path = checkpoint_path_for_training_end(
        tmp_path,
        train_end_month="202606",
        as_of_date="2026-07-31",
    )

    assert path == tmp_path / "checkpoint_2026Q2_asof_20260731.pt"


def test_later_signal_month_reuses_quarter_checkpoint_availability_date(tmp_path: Path):
    path = checkpoint_path_for_training_end(
        tmp_path,
        train_end_month="202606",
        as_of_date="2026-08-31",
    )

    assert path == tmp_path / "checkpoint_2026Q2_asof_20260731.pt"


def test_checkpoint_round_trip_preserves_deterministic_scores(tmp_path: Path):
    samples = {"2026-05": _sample("2026-05"), "2026-07": _sample("2026-07")}
    labels = pd.DataFrame(
        [
            {"month": "2026-05", "code": "sz.000001", "future_excess_return_1m": 0.1},
            {"month": "2026-05", "code": "sz.000002", "future_excess_return_1m": -0.1},
        ]
    )
    model = train_final_model(
        monthly_samples=samples,
        labels_df=labels,
        train_months=["2026-05"],
        epochs=1,
        seed=7,
        fusion_type="cross_encoder",
        graph_module_type="relation_aware",
    )
    metadata = build_checkpoint_metadata(
        model=model,
        train_months=["2026-05"],
        train_start_month="202605",
        train_end_month="202605",
        as_of_date="2026-07-31",
        signal_month="202607",
        seed=7,
        fusion_type="cross_encoder",
        graph_module_type="relation_aware",
        edge_type_prior_mode="none",
        training_objective="ranknet_pairwise",
        training_label="future_excess_return_1m",
        edge_type_mapping={"market_corr_topk": 0},
        industry_mapping={"银行": 0, "电子": 1},
    )
    checkpoint_path = tmp_path / "model.pt"
    save_prediction_checkpoint(model, metadata, checkpoint_path)

    loaded_model, loaded_metadata = load_prediction_checkpoint(
        checkpoint_path,
        expected={
            "feature_schema_version": metadata["feature_schema_version"],
            "fusion_type": "cross_encoder",
            "graph_module_type": "relation_aware",
            "edge_type_mapping": {"market_corr_topk": 0},
        },
    )

    first = score_month(loaded_model, samples["2026-07"])
    second = score_month(loaded_model, samples["2026-07"])
    pd.testing.assert_frame_equal(first, second)
    assert loaded_metadata["train_months"] == ["2026-05"]
    assert loaded_model.training is False


def test_checkpoint_loader_rejects_schema_mismatch(tmp_path: Path):
    model = DualChannelSelector(
        node_feature_dim=2,
        hyperedge_feature_dim=3,
        num_industries=2,
        graph_module_type="relation_aware",
    )
    metadata = build_checkpoint_metadata(
        model=model,
        train_months=["2026-05"],
        train_start_month="202605",
        train_end_month="202605",
        as_of_date="2026-07-31",
        signal_month="202607",
        seed=7,
        fusion_type="cross_encoder",
        graph_module_type="relation_aware",
        edge_type_prior_mode="none",
        training_objective="ranknet_pairwise",
        training_label="future_excess_return_1m",
        edge_type_mapping={"market_corr_topk": 0},
        industry_mapping={"银行": 0, "电子": 1},
    )
    checkpoint_path = tmp_path / "model.pt"
    save_prediction_checkpoint(model, metadata, checkpoint_path)

    with pytest.raises(ValueError, match="feature_schema_version"):
        load_prediction_checkpoint(
            checkpoint_path,
            expected={"feature_schema_version": "unexpected-schema"},
        )


def test_signal_month_must_not_be_in_training_range():
    samples = {"2026-07": _sample("2026-07")}
    labels = pd.DataFrame(
        [{"month": "2026-07", "code": "sz.000001", "future_excess_return_1m": 0.1}]
    )

    with pytest.raises(ValueError, match="before signal_month"):
        select_training_months(
            samples,
            labels,
            signal_month="202607",
            train_start_month="202607",
            train_end_month="202607",
            training_label="future_excess_return_1m",
        )


def test_training_range_rejects_months_without_labels():
    samples = {
        "2026-05": _sample("2026-05"),
        "2026-06": _sample("2026-06"),
        "2026-07": _sample("2026-07"),
    }
    labels = pd.DataFrame(
        [{"month": "2026-06", "code": "sz.000001", "future_excess_return_1m": 0.1}]
    )

    with pytest.raises(ValueError, match="missing labels.*2026-05"):
        select_training_months(
            samples,
            labels,
            signal_month="202607",
            train_start_month="202605",
            train_end_month="202606",
            training_label="future_excess_return_1m",
        )


def test_month_without_active_reports_keeps_embedding_dimension():
    monthly_features = pd.DataFrame(
        [{"month": "2026-07", "code": "sz.000001", "ret_20d": 0.1}]
    )
    reports = pd.DataFrame(
        [{"report_id": "old", "report_month": "2026-01", "publish_date": "2026-01-02"}]
    )
    embeddings = pd.DataFrame(
        [{"report_id": "old", "emb_0": 0.1, "emb_1": 0.2, "emb_2": 0.3}]
    )

    samples = build_monthly_samples(
        monthly_features_df=monthly_features,
        reports_df=reports,
        incidence_df=pd.DataFrame(columns=["report_id", "stock_code", "incidence_weight"]),
        report_embeddings_df=embeddings,
        stock_graph_edges_df=pd.DataFrame(),
        decay_months=3,
    )

    assert samples["2026-07"]["hyperedge_features"].shape == (0, 3)
