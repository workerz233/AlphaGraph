from __future__ import annotations

from pathlib import Path
import tomllib

import pandas as pd
import pytest

from alphagraph.gnn import predict
from alphagraph.pipeline_config import resolve_config


def test_build_recommendations_applies_filters_and_stable_ranking():
    scores = pd.DataFrame(
        [
            {"month": "2026-07", "code": "sz.000001", "score": 0.9},
            {"month": "2026-07", "code": "sz.000002", "score": 0.8},
            {"month": "2026-07", "code": "sz.000003", "score": 0.8},
            {"month": "2026-07", "code": "sz.000004", "score": 0.7},
        ]
    )
    features = pd.DataFrame(
        [
            {"month": "2026-07", "code": "sz.000001", "industry": "银行", "is_tradable": True, "is_st": 1, "month_end_close": 10.0, "amount_20d_mean": 100.0},
            {"month": "2026-07", "code": "sz.000002", "industry": "电子", "is_tradable": True, "is_st": 0, "month_end_close": 10.0, "amount_20d_mean": 90.0},
            {"month": "2026-07", "code": "sz.000003", "industry": "计算机", "is_tradable": True, "is_st": 0, "month_end_close": 10.0, "amount_20d_mean": 80.0},
            {"month": "2026-07", "code": "sz.000004", "industry": "银行", "is_tradable": True, "is_st": 0, "month_end_close": 1.0, "amount_20d_mean": 70.0},
        ]
    )
    stock_basic = pd.DataFrame(
        [
            {"code": "sz.000002", "code_name": "股票二"},
            {"code": "sz.000003", "code_name": "股票三"},
        ]
    )

    result = predict.build_recommendations(
        scores,
        features,
        stock_basic_df=stock_basic,
        topk=20,
        liquidity_quantile=0.0,
        min_price=2.0,
        exclude_st=True,
        as_of_date="2026-07-31",
    )

    assert result[["rank", "code", "name", "industry"]].to_dict(orient="records") == [
        {"rank": 1, "code": "sz.000002", "name": "股票二", "industry": "电子"},
        {"rank": 2, "code": "sz.000003", "name": "股票三", "industry": "计算机"},
    ]
    assert result["execution_from"].unique().tolist() == ["2026-08-03"]


def test_force_retrain_uses_versioned_path_instead_of_overwriting(tmp_path: Path):
    canonical = tmp_path / "checkpoint_2026Q2_asof_20260731.pt"
    canonical.write_bytes(b"existing")

    selected = predict.resolve_checkpoint_output_path(
        canonical,
        force_retrain=True,
        run_tag="20260804_120000",
    )

    assert selected == tmp_path / "checkpoint_2026Q2_asof_20260731_rerun_20260804_120000.pt"
    assert canonical.read_bytes() == b"existing"


def test_prediction_cli_forwards_server_arguments(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_run_prediction(**kwargs):
        captured.update(kwargs)
        return {"recommendations_csv": tmp_path / "recommendations_202607.csv"}

    monkeypatch.setattr(predict, "run_prediction", fake_run_prediction)

    predict.main(
        [
            "--artifact-dir",
            str(tmp_path),
            "--signal-month",
            "202607",
            "--as-of-date",
            "2026-07-31",
            "--train-start-month",
            "202507",
            "--train-end-month",
            "202606",
            "--checkpoint-cadence-months",
            "3",
            "--topk",
            "20",
            "--force-retrain",
        ]
    )

    assert captured["signal_month"] == "202607"
    assert captured["train_start_month"] == "202507"
    assert captured["train_end_month"] == "202606"
    assert captured["checkpoint_cadence_months"] == 3
    assert captured["force_retrain"] is True


def test_server_scripts_expose_prediction_contract():
    prediction_script = Path("alphagraph/run_prediction.sh").read_text(encoding="utf-8")
    server_script = Path("scripts/run_202607_prediction.sh").read_text(encoding="utf-8")
    kg_script = Path("alphagraph/build_kg.sh").read_text(encoding="utf-8")

    assert "--data-start-month" in prediction_script
    assert "--as-of-date" in prediction_script
    assert "--checkpoint-cadence-months" in prediction_script
    assert 'conda run --no-capture-output -n "$ENV_NAME" python' in prediction_script
    assert "TRAIN_START_MONTH=\"${TRAIN_START_MONTH:-202507}\"" in server_script
    assert "FORCE_RETRAIN=true" in server_script
    assert "2>&1 | tee \"$RUN_LOG\"" in server_script
    assert "python3" not in kg_script
    assert 'conda run -n "$ENV_NAME" python' in kg_script
    assert "--llm-kg-api-key" not in kg_script


def test_llm_api_key_is_injected_from_server_environment(monkeypatch):
    with Path("alphagraph/pipeline_config.toml").open("rb") as handle:
        config = tomllib.load(handle)

    assert config["defaults"]["llm_kg_api_key"] == ""
    monkeypatch.setenv("LLM_KG_API_KEY", "server-secret")
    values = resolve_config(config, mode="kg", config_name="phase1_relation_aware")
    assert values["LLM_KG_API_KEY"] == "server-secret"


def test_run_prediction_writes_checkpoint_scores_and_recommendations(tmp_path: Path):
    codes = ["sz.000001", "sz.000002"]
    features = pd.DataFrame(
        [
            {
                "month": month,
                "date": pd.Timestamp(f"{month}-01") + pd.offsets.MonthEnd(0),
                "code": code,
                "industry": "银行" if index == 0 else "电子",
                "industry_id": index,
                "ret_20d": 0.1 - index * 0.1,
                "is_tradable": True,
                "is_st": 0,
                "month_end_close": 10.0,
                "amount_20d_mean": 100.0 - index,
            }
            for month in ("2026-06", "2026-07")
            for index, code in enumerate(codes)
        ]
    )
    frames = {
        "reports.parquet": pd.DataFrame(
            [{"report_id": "r1", "report_month": "2026-06", "publish_date": "2026-06-10"}]
        ),
        "incidence.parquet": pd.DataFrame(
            [
                {"report_id": "r1", "stock_code": code, "incidence_weight": 1.0}
                for code in codes
            ]
        ),
        "monthly_stock_features.parquet": features,
        "monthly_labels.parquet": pd.DataFrame(
            [
                {"month": "2026-06", "code": codes[0], "future_excess_return_1m": 0.1},
                {"month": "2026-06", "code": codes[1], "future_excess_return_1m": -0.1},
            ]
        ),
        "report_embeddings.parquet": pd.DataFrame(
            [{"report_id": "r1", "emb_0": 0.1, "emb_1": 0.2, "emb_2": 0.3}]
        ),
        "stock_graph_edges.parquet": pd.DataFrame(
            [
                {
                    "month": month,
                    "src_code": codes[0],
                    "dst_code": codes[1],
                    "edge_type": "market_corr_topk",
                    "edge_weight": 1.0,
                    "source_id": f"market:{month}",
                    "report_id": "",
                }
                for month in ("2026-06", "2026-07")
            ]
        ),
        "stock_basic.parquet": pd.DataFrame(
            [{"code": code, "code_name": f"股票{index + 1}"} for index, code in enumerate(codes)]
        ),
    }
    for filename, frame in frames.items():
        frame.to_parquet(tmp_path / filename, index=False)

    paths = predict.run_prediction(
        artifact_dir=tmp_path,
        signal_month="202607",
        as_of_date="2026-07-31",
        train_start_month="202606",
        train_end_month="202606",
        epochs=1,
        topk=1,
        liquidity_quantile=0.0,
        min_price=0.0,
        run_tag="test-run",
    )

    assert paths["checkpoint"].name == "checkpoint_2026Q2_asof_20260731.pt"
    assert pd.read_parquet(paths["scores"])["month"].unique().tolist() == ["2026-07"]
    assert len(pd.read_csv(paths["recommendations_csv"])) == 1
    metadata = __import__("json").loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["train_months"] == ["2026-06"]
    assert metadata["signal_month"] == "2026-07"

    with pytest.raises(ValueError, match="Checkpoint epochs mismatch"):
        predict.run_prediction(
            artifact_dir=tmp_path,
            signal_month="202607",
            as_of_date="2026-07-31",
            train_start_month="202606",
            train_end_month="202606",
            epochs=2,
            topk=1,
            liquidity_quantile=0.0,
            min_price=0.0,
            run_tag="incompatible-epochs",
        )
