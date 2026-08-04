from __future__ import annotations

import inspect

import pandas as pd

from alphagraph.backtest import runner


def test_backtest_report_template_contains_responsive_overflow_guards():
    source = inspect.getsource(runner._write_html_report)

    assert "overflow-x: hidden;" in source
    assert "max-width: 100%;" in source
    assert ".ant-card, .ant-card-body" in source
    assert ".ant-table-wrapper" in source
    assert "overflow-wrap: anywhere;" in source
    assert "scroll: {{ x: 'max-content'" in source


def test_backtest_report_overview_includes_sharpe_ratio():
    source = inspect.getsource(runner._write_html_report)

    assert "夏普比率(年化)" in source
    assert "overviewMetricNames" in source
    assert "metricRows.filter((row) => overviewMetricNames.includes(row.name))" in source


def test_backtest_report_includes_experiment_metadata():
    source = inspect.getsource(runner._write_html_report)

    assert "实验项目:" in source
    assert "experimentDescription" in source


def test_generated_backtest_html_displays_experiment_metadata(tmp_path):
    output_path = tmp_path / "report.html"

    runner._write_html_report(
        backtest_df=pd.DataFrame(),
        metrics={"months": 0},
        output_path=output_path,
        scores_path=tmp_path / "monthly_scores_phase1_relation_aware.parquet",
        labels_path=tmp_path / "monthly_labels.parquet",
        benchmark_path=tmp_path / "monthly_benchmark_returns.parquet",
        output_returns_path=tmp_path / "backtest_returns.parquet",
        output_metrics_path=tmp_path / "backtest_metrics.json",
        model_runs_dir=tmp_path / "model_runs",
        run_config={
            "config_name": "phase1_relation_aware",
            "experiment_name": "关系感知图聚合主模型",
            "experiment_description": "cross_encoder 融合 + 关系感知图聚合",
        },
    )

    html = output_path.read_text(encoding="utf-8")
    assert "实验项目: ${experimentName}" in html
    assert 'const experimentName = "关系感知图聚合主模型";' in html
    assert 'const experimentDescription = "cross_encoder 融合 + 关系感知图聚合";' in html


def test_backtest_index_template_contains_responsive_overflow_guards():
    source = inspect.getsource(runner._write_backtest_index_html)

    assert "overflow-x: hidden;" in source
    assert "max-width: 100%;" in source
    assert ".ant-card, .ant-card-body" in source
    assert ".ant-table-wrapper" in source
    assert "scroll: {{ x: 'max-content'" in source


def test_backtest_index_displays_experiment_metadata():
    source = inspect.getsource(runner._write_backtest_index_html)

    assert "title: '实验项目'" in source
    assert "title: '实验说明'" in source
