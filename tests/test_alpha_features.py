from pathlib import Path

import pandas as pd
import pytest

from alphagraph.data.alpha_features import (
    ALPHA_FEATURE_COLUMNS,
    EXPECTED_ALPHA_FACTORS,
    build_monthly_alpha_features,
    load_alpha_factor_manifest,
)


def _ranking_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor": [factor for factor, _ in EXPECTED_ALPHA_FACTORS],
            "direction": [direction for _, direction in EXPECTED_ALPHA_FACTORS],
            "validation_rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
        }
    )


def test_load_alpha_factor_manifest_validates_fixed_top10(tmp_path: Path):
    ranking = _ranking_frame()
    ranking.to_csv(tmp_path / "factor_ranking.csv", index=False)

    manifest = load_alpha_factor_manifest(tmp_path)

    assert manifest["factors"] == [factor for factor, _ in EXPECTED_ALPHA_FACTORS]
    assert manifest["directions"] == {factor: direction for factor, direction in EXPECTED_ALPHA_FACTORS}
    assert manifest["columns"] == list(ALPHA_FEATURE_COLUMNS)


def test_load_alpha_factor_manifest_rejects_changed_ranking(tmp_path: Path):
    ranking = _ranking_frame()
    ranking.loc[0, "factor"] = "A158.KUP"
    ranking.to_csv(tmp_path / "factor_ranking.csv", index=False)

    with pytest.raises(ValueError, match="fixed top-10"):
        load_alpha_factor_manifest(tmp_path)


def test_load_alpha_factor_manifest_rejects_changed_rank_values(tmp_path: Path):
    ranking = _ranking_frame()
    ranking.loc[9, "validation_rank"] = 10
    ranking.to_csv(tmp_path / "factor_ranking.csv", index=False)

    with pytest.raises(ValueError, match="fixed top-10"):
        load_alpha_factor_manifest(tmp_path)


def test_build_monthly_alpha_features_uses_month_end_and_direction(tmp_path: Path, monkeypatch):
    ranking = _ranking_frame()
    ranking.to_csv(tmp_path / "factor_ranking.csv", index=False)
    # Two dates in one month; the later value must be selected.  The first
    # factor has direction -1, so its value must be sign-aligned.
    daily = pd.DataFrame(
        [
            {"date": "2025-01-30", "code": "sz.000001", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100, "amount": 1000},
            {"date": "2025-01-31", "code": "sz.000001", "open": 10.5, "high": 12, "low": 10, "close": 11.5, "volume": 110, "amount": 1200},
        ]
    )
    monthly = pd.DataFrame(
        [{"month": "2025-01", "date": "2025-01-31", "code": "sz.000001"},
         {"month": "2025-01", "date": "2025-01-31", "code": "sz.000002"}]
    )

    # Keep the test independent of the reference formula implementations.
    from alphagraph.data import alpha_features

    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-30", "2025-01-31"]),
            "asset": ["000001", "000001"],
            **{factor: [float(i), float(i + 1)] for i, (factor, _) in enumerate(EXPECTED_ALPHA_FACTORS)},
        }
    ).set_index(["date", "asset"])
    monkeypatch.setattr(alpha_features, "_compute_daily_alpha_values", lambda frame, factors: raw)

    features, audit, manifest = build_monthly_alpha_features(daily, monthly, tmp_path)

    first_column = ALPHA_FEATURE_COLUMNS[0]
    assert features.loc[features["code"] == "sz.000001", first_column].iat[0] == -(0 + 1)
    assert features.loc[features["code"] == "sz.000002", first_column].iat[0] == 0
    assert audit["coverage"].min() == 0.5
    assert manifest["aggregation"] == "month_end_last_valid"


def test_build_monthly_alpha_features_handles_multiple_stocks(tmp_path: Path, monkeypatch):
    _ranking_frame().to_csv(tmp_path / "factor_ranking.csv", index=False)
    daily = pd.DataFrame(
        [
            {"date": "2025-01-30", "code": "sz.000001", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
            {"date": "2025-01-29", "code": "sz.000002", "open": 20, "high": 21, "low": 19, "close": 20, "volume": 200},
        ]
    )
    monthly = pd.DataFrame(
        [
            {"month": "2025-01", "date": "2025-01-31", "code": "sz.000001"},
            {"month": "2025-01", "date": "2025-01-31", "code": "sz.000002"},
        ]
    )
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-30"), "000001"), (pd.Timestamp("2025-01-29"), "000002")],
        names=["date", "asset"],
    )
    raw = pd.DataFrame(
        {factor: [float(i), float(i + 10)] for i, (factor, _) in enumerate(EXPECTED_ALPHA_FACTORS)},
        index=index,
    )
    from alphagraph.data import alpha_features

    monkeypatch.setattr(alpha_features, "_compute_daily_alpha_values", lambda frame, factors: raw)

    features, _, _ = build_monthly_alpha_features(daily, monthly, tmp_path)

    assert features["code"].tolist() == ["sz.000001", "sz.000002"]


def test_node_feature_schema_includes_exactly_ten_alpha_columns():
    from alphagraph.model.core import NODE_FEATURE_COLUMNS

    assert len(ALPHA_FEATURE_COLUMNS) == 10
    assert len(NODE_FEATURE_COLUMNS) == 41
    assert NODE_FEATURE_COLUMNS[-10:] == list(ALPHA_FEATURE_COLUMNS)


def test_build_monthly_samples_exposes_41_node_features():
    import numpy as np
    from alphagraph.model.core import build_monthly_samples

    features = pd.DataFrame(
        [{"month": "2025-01", "code": "sz.000001", "date": "2025-01-31"}]
    )
    samples = build_monthly_samples(
        monthly_features_df=features,
        reports_df=pd.DataFrame(columns=["report_id", "report_month"]),
        incidence_df=pd.DataFrame(columns=["report_id", "stock_code"]),
        report_embeddings_df=pd.DataFrame(columns=["report_id"]),
        stock_graph_edges_df=pd.DataFrame(),
    )

    assert samples["2025-01"]["node_features"].shape == (1, 41)
    assert np.isfinite(samples["2025-01"]["node_features"]).all()


def test_build_monthly_alpha_features_preserves_unsorted_monthly_rows(tmp_path: Path, monkeypatch):
    _ranking_frame().to_csv(tmp_path / "factor_ranking.csv", index=False)
    daily = pd.DataFrame(
        [
            {"date": "2025-01-30", "code": "sz.000001", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
            {"date": "2025-01-30", "code": "sz.000002", "open": 20, "high": 21, "low": 19, "close": 20, "volume": 200},
        ]
    )
    monthly = pd.DataFrame(
        [
            {"month": "2025-01", "date": "2025-01-31", "code": "sz.000002"},
            {"month": "2025-01", "date": "2025-01-31", "code": "sz.000001"},
        ]
    )
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-30"), "000001"), (pd.Timestamp("2025-01-30"), "000002")],
        names=["date", "asset"],
    )
    raw = pd.DataFrame(
        {factor: [1.0, 2.0] for factor, _ in EXPECTED_ALPHA_FACTORS}, index=index
    )
    from alphagraph.data import alpha_features

    monkeypatch.setattr(alpha_features, "_compute_daily_alpha_values", lambda frame, factors: raw)
    features, _, _ = build_monthly_alpha_features(daily, monthly, tmp_path)

    assert features[ALPHA_FEATURE_COLUMNS[1]].tolist() == [2.0, 1.0]


def test_build_monthly_alpha_features_uses_each_factors_last_valid_value(tmp_path: Path, monkeypatch):
    _ranking_frame().to_csv(tmp_path / "factor_ranking.csv", index=False)
    daily = pd.DataFrame(
        [
            {"date": "2025-01-30", "code": "sz.000001", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
            {"date": "2025-01-31", "code": "sz.000001", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
        ]
    )
    monthly = pd.DataFrame(
        [{"month": "2025-01", "date": "2025-01-31", "code": "sz.000001"}]
    )
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-30"), "000001"), (pd.Timestamp("2025-01-31"), "000001")],
        names=["date", "asset"],
    )
    raw = pd.DataFrame(
        {factor: [3.0, float("nan")] for factor, _ in EXPECTED_ALPHA_FACTORS}, index=index
    )
    from alphagraph.data import alpha_features

    monkeypatch.setattr(alpha_features, "_compute_daily_alpha_values", lambda frame, factors: raw)
    features, audit, _ = build_monthly_alpha_features(daily, monthly, tmp_path)

    assert features[ALPHA_FEATURE_COLUMNS[1]].iat[0] == 3.0
    assert audit["coverage"].min() == 1.0
