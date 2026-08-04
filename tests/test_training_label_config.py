from __future__ import annotations

import tomllib

import pandas as pd
import pytest

from alphagraph.gnn.train import validate_training_scores
from alphagraph.model.core import build_training_label_lookup
from alphagraph.pipeline_config import resolve_config


def test_build_training_label_lookup_uses_selected_label_column():
    labels = pd.DataFrame(
        [
            {
                "month": "2026-01",
                "code": "sz.000001",
                "future_excess_return_1m": 0.10,
                "future_excess_sharpe_1m": 1.25,
            }
        ]
    )

    lookup = build_training_label_lookup(labels, "future_excess_sharpe_1m")

    assert lookup == {("2026-01", "sz.000001"): 1.25}


def test_training_label_profile_exports_sharpe_label():
    with open("alphagraph/pipeline_config.toml", "rb") as handle:
        config = tomllib.load(handle)

    values = resolve_config(config, mode="pipeline", config_name="phase1_relation_aware_sharpe_label")

    assert values["TRAINING_LABEL"] == "future_excess_sharpe_1m"


def test_ablation_profile_exports_experiment_metadata():
    with open("alphagraph/pipeline_config.toml", "rb") as handle:
        config = tomllib.load(handle)

    values = resolve_config(config, mode="pipeline", config_name="phase1_relation_aware")

    assert values["EXPERIMENT_NAME"] == "关系感知图聚合主模型"
    assert values["EXPERIMENT_DESCRIPTION"] == "cross_encoder 融合 + 关系感知图聚合"


def test_label_ablation_group_contains_return_and_sharpe_labels():
    with open("alphagraph/pipeline_config.toml", "rb") as handle:
        config = tomllib.load(handle)

    assert config["profile_groups"]["label_ablation"] == [
        "phase1_relation_aware",
        "phase1_relation_aware_sharpe_label",
    ]


def test_multimonth_config_uses_markdown_root(monkeypatch):
    monkeypatch.delenv("MARKDOWN_DIR", raising=False)
    config = {
        "defaults": {
            "markdown_dir": "reportdata/202601_equity_industry_markdown",
            "markdown_root": "reportdata",
            "start_month": "202503",
            "end_month": "202601",
        }
    }

    values = resolve_config(config, mode="graph")

    assert values["MARKDOWN_DIR"] == "reportdata"


def test_single_month_config_keeps_markdown_dir(monkeypatch):
    monkeypatch.delenv("MARKDOWN_DIR", raising=False)
    config = {
        "defaults": {
            "markdown_dir": "reportdata/202601_equity_industry_markdown",
            "markdown_root": "reportdata",
            "start_month": "202601",
            "end_month": "202601",
        }
    }

    values = resolve_config(config, mode="graph")

    assert values["MARKDOWN_DIR"] == "reportdata/202601_equity_industry_markdown"


def test_explicit_markdown_dir_overrides_multimonth_root(monkeypatch):
    monkeypatch.setenv("MARKDOWN_DIR", "custom/reports")
    config = {
        "defaults": {
            "markdown_dir": "reportdata/202601_equity_industry_markdown",
            "markdown_root": "reportdata",
            "start_month": "202503",
            "end_month": "202601",
        }
    }

    values = resolve_config(config, mode="graph")

    assert values["MARKDOWN_DIR"] == "custom/reports"


def test_empty_training_scores_raise_multimonth_diagnostic():
    with pytest.raises(RuntimeError, match="至少需要 2 个有效月份"):
        validate_training_scores(
            pd.DataFrame(columns=["month", "code", "score"]),
            monthly_samples={"2026-01": {}},
        )
