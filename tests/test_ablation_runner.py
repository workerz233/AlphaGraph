from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts import run_alphagraph_ablation as ablation_runner
from scripts.run_alphagraph_ablation import (
    baseline_config_overrides,
    collect_latest_results,
    default_dataset_name,
    execution_env,
    load_ablation_runner_config,
    resolve_dataset_dir,
    resolve_profiles,
    run_experiment_sequence,
    run_profiles,
    write_summary_files,
)


def test_shell_wrapper_configures_and_runs_one_profile(tmp_path):
    project_root = _write_shell_wrapper_fixture(tmp_path)
    capture_path = tmp_path / "conda_args.txt"
    env = _shell_wrapper_test_env(tmp_path, project_root, capture_path)

    result = subprocess.run(
        [
            "bash",
            str(Path.cwd() / "scripts" / "run_ablation_target.sh"),
            "profile_b",
        ],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    config_text = (project_root / "alphagraph" / "pipeline_config.toml").read_text(
        encoding="utf-8"
    )
    assert 'target = "profile_b"' in config_text
    assert "dry_run = false" in config_text
    assert "collect_only = false" in config_text
    assert capture_path.read_text(encoding="utf-8").splitlines() == [
        "run",
        "--no-capture-output",
        "-n",
        "rGNN",
        "python",
        "-u",
        "scripts/run_alphagraph_ablation.py",
    ]


def test_shell_wrapper_rejects_profile_group(tmp_path):
    project_root = _write_shell_wrapper_fixture(tmp_path)
    capture_path = tmp_path / "conda_args.txt"
    env = _shell_wrapper_test_env(tmp_path, project_root, capture_path)

    result = subprocess.run(
        [
            "bash",
            str(Path.cwd() / "scripts" / "run_ablation_target.sh"),
            "pair_group",
        ],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "单个 profile" in result.stderr
    assert not capture_path.exists()


def test_main_runs_from_toml_without_cli_arguments(tmp_path, monkeypatch):
    config_dir = tmp_path / "alphagraph"
    config_dir.mkdir()
    (config_dir / "pipeline_config.toml").write_text(
        """
[defaults]
artifact_namespace = "alphagraph"
env_name = "rGNN"
epochs = 20
topk = 20
cost_bps = 1

[ablation_runner]
target = "phase1_relation_aware"
start_month = "202503"
end_month = "202601"
dataset_name = "202503_202601"
dry_run = true
collect_only = false

[profiles.phase1_relation_aware]
""".strip(),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        ablation_runner,
        "__file__",
        str(tmp_path / "scripts" / "run_alphagraph_ablation.py"),
    )
    monkeypatch.setattr(
        ablation_runner,
        "run_experiment_sequence",
        lambda **kwargs: captured.update(kwargs),
    )

    assert ablation_runner.main() == 0
    assert captured["profiles"] == ["phase1_relation_aware"]
    assert captured["env_overrides"] == {
        "START_MONTH": "202503",
        "END_MONTH": "202601",
        "DATASET_NAME": "202503_202601",
        "EPOCHS": "20",
        "TOPK": "20",
        "COST_BPS": "1",
    }
    assert captured["dry_run"] is True
    assert captured["collect_only"] is False


def test_load_ablation_runner_config_reads_repository_execution_section():
    runner_config = load_ablation_runner_config(Path.cwd())

    assert runner_config.target == "phase1_relation_aware"
    assert runner_config.start_month == "202503"
    assert runner_config.end_month == "202601"
    assert runner_config.dataset_name == "202503_202601"
    assert runner_config.dry_run is False
    assert runner_config.collect_only is False


def test_resolve_profiles_accepts_single_profile():
    config = {
        "profiles": {"phase1_relation_aware": {}},
        "profile_groups": {"objective_ablation": ["phase1_relation_aware"]},
    }

    assert resolve_profiles(config, "phase1_relation_aware") == ["phase1_relation_aware"]


def test_resolve_profiles_uses_configured_group_members():
    config = {
        "profiles": {
            "phase1_relation_aware": {},
            "phase1_relation_aware_mse_baseline": {},
        },
        "profile_groups": {
            "objective_ablation": [
                "phase1_relation_aware",
                "phase1_relation_aware_mse_baseline",
            ]
        },
    }

    assert resolve_profiles(config, "objective_ablation") == [
        "phase1_relation_aware",
        "phase1_relation_aware_mse_baseline",
    ]


def test_resolve_profiles_rejects_unknown_target():
    config = {"profiles": {"phase1_relation_aware": {}}, "profile_groups": {}}

    with pytest.raises(ValueError, match="Unknown ablation target"):
        resolve_profiles(config, "missing_profile")


def test_load_ablation_runner_config_rejects_conflicting_modes(tmp_path):
    config_dir = tmp_path / "alphagraph"
    config_dir.mkdir()
    (config_dir / "pipeline_config.toml").write_text(
        """
[defaults]
artifact_namespace = "alphagraph"
env_name = "rGNN"

[ablation_runner]
target = "phase1_relation_aware"
start_month = "202503"
end_month = "202601"
dataset_name = "202503_202601"
dry_run = true
collect_only = true

[profiles.phase1_relation_aware]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dry_run.*collect_only"):
        load_ablation_runner_config(tmp_path)


def test_run_profiles_dry_run_does_not_execute_commands(tmp_path):
    calls: list[tuple[list[str], Path, dict[str, str]]] = []

    run_profiles(
        profiles=["phase1_relation_aware", "graph_mean_baseline"],
        project_root=tmp_path,
        dry_run=True,
        runner=lambda command, cwd: calls.append(command),
    )

    assert calls == []


def test_run_profiles_passes_temporary_window_env_to_pipeline(tmp_path):
    calls: list[tuple[list[str], Path, dict[str, str]]] = []

    run_profiles(
        profiles=["phase1_relation_aware"],
        project_root=tmp_path,
        dry_run=False,
        env_overrides={
            "START_MONTH": "202503",
            "END_MONTH": "202601",
            "DATASET_NAME": "202503_202601",
        },
        runner=lambda command, cwd, env: calls.append((command, cwd, env)),
    )

    assert calls == [
        (
            ["bash", "alphagraph/run_pipeline.sh", "phase1_relation_aware"],
            tmp_path,
            {
                "START_MONTH": "202503",
                "END_MONTH": "202601",
                "DATASET_NAME": "202503_202601",
            },
        )
    ]


def test_default_dataset_name_and_dir_use_temporary_window(tmp_path):
    config_dir = tmp_path / "alphagraph"
    config_dir.mkdir()
    (config_dir / "pipeline_config.toml").write_text(
        '[defaults]\nartifact_namespace = "alphagraph"\ndataset_name = "202401_202601"\n',
        encoding="utf-8",
    )

    assert default_dataset_name("202503", "202601") == "202503_202601"
    assert (
        resolve_dataset_dir(
            project_root=tmp_path,
            artifact_namespace=None,
            dataset_name=None,
            start_month="202503",
            end_month="202601",
        )
        == tmp_path / "artifacts" / "alphagraph" / "202503_202601"
    )


def test_execution_env_sets_month_window_and_dataset():
    env = execution_env(
        start_month="202503",
        end_month="202601",
        dataset_name=None,
        baseline_overrides={},
    )

    assert env == {
        "START_MONTH": "202503",
        "END_MONTH": "202601",
        "DATASET_NAME": "202503_202601",
    }


def test_baseline_config_overrides_lock_core_experiment_settings(tmp_path):
    config_dir = tmp_path / "alphagraph"
    config_dir.mkdir()
    (config_dir / "pipeline_config.toml").write_text(
        "[defaults]\nepochs = 20\ntopk = 20\ncost_bps = 1\n",
        encoding="utf-8",
    )

    assert baseline_config_overrides(tmp_path) == {
        "EPOCHS": "20",
        "TOPK": "20",
        "COST_BPS": "1",
    }


def test_execution_env_includes_locked_baseline_values():
    env = execution_env(
        start_month="202503",
        end_month="202601",
        dataset_name=None,
        baseline_overrides={
            "EPOCHS": "20",
            "TOPK": "20",
            "COST_BPS": "1",
        },
    )

    assert env["EPOCHS"] == "20"
    assert env["TOPK"] == "20"
    assert env["COST_BPS"] == "1"


def test_run_experiment_sequence_writes_summary_after_each_profile(tmp_path):
    calls: list[tuple[list[str], Path, dict[str, str]]] = []
    written_profiles: list[list[str]] = []

    run_experiment_sequence(
        profiles=["phase1_relation_aware", "graph_mean_baseline"],
        project_root=tmp_path,
        dataset_dir=tmp_path / "artifacts" / "alphagraph" / "202503_202601",
        env_overrides={"START_MONTH": "202503"},
        dry_run=False,
        collect_only=False,
        runner=lambda command, cwd, env: calls.append((command, cwd, env)),
        collector=lambda dataset_dir, profiles: [{"config_name": profile} for profile in profiles],
        writer=lambda dataset_dir, rows: written_profiles.append(
            [str(row["config_name"]) for row in rows]
        ),
    )

    assert [call[0][-1] for call in calls] == ["phase1_relation_aware", "graph_mean_baseline"]
    assert written_profiles == [
        ["phase1_relation_aware"],
        ["phase1_relation_aware", "graph_mean_baseline"],
    ]


def test_collect_latest_results_filters_by_profile_and_writes_summary(tmp_path):
    dataset_dir = tmp_path / "artifacts" / "alphagraph" / "202503_202601"
    first_run = dataset_dir / "backtest_runs" / "2026_07_01-010101"
    latest_run = dataset_dir / "backtest_runs" / "2026_07_02-010101"
    other_run = dataset_dir / "backtest_runs" / "2026_07_03-010101"
    first_run.mkdir(parents=True)
    latest_run.mkdir(parents=True)
    other_run.mkdir(parents=True)
    _write_metrics(first_run / "backtest_metrics.json", "phase1_relation_aware", 0.01)
    _write_metrics(latest_run / "backtest_metrics.json", "phase1_relation_aware", 0.02)
    _write_metrics(other_run / "backtest_metrics.json", "graph_mean_baseline", -0.01)

    rows = collect_latest_results(
        dataset_dir=dataset_dir,
        profiles=["phase1_relation_aware", "graph_mean_baseline"],
    )

    assert [row["config_name"] for row in rows] == [
        "phase1_relation_aware",
        "graph_mean_baseline",
    ]
    assert rows[0]["run_id"] == "2026_07_02-010101"
    assert rows[0]["portfolio_total_return"] == 0.02

    csv_path, md_path = write_summary_files(dataset_dir=dataset_dir, rows=rows)

    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("config_name,run_id")
    assert "| phase1_relation_aware | 2026_07_02-010101 |" in md_path.read_text(
        encoding="utf-8"
    )
    assert "关系感知图聚合主模型" in md_path.read_text(encoding="utf-8")
    assert "experiment_name" in csv_path.read_text(encoding="utf-8").splitlines()[0]


def _write_metrics(path: Path, config_name: str, portfolio_total_return: float) -> None:
    path.write_text(
        json.dumps(
            {
                "months": 24,
                "portfolio_total_return": portfolio_total_return,
                "excess_hs300_total_return": portfolio_total_return - 0.01,
                "excess_zz500_total_return": portfolio_total_return - 0.02,
                "mean_ic": 0.03,
                "sharpe_ratio": 1.2,
                "average_turnover": 0.5,
                "run_config": {
                    "config_name": config_name,
                    "experiment_name": "关系感知图聚合主模型",
                    "experiment_description": "cross_encoder 融合 + 关系感知图聚合",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_shell_wrapper_fixture(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    config_dir = project_root / "alphagraph"
    scripts_dir = project_root / "scripts"
    config_dir.mkdir(parents=True)
    scripts_dir.mkdir()
    (config_dir / "pipeline_config.toml").write_text(
        """
[defaults]
artifact_namespace = "alphagraph"
env_name = "rGNN"

[ablation_runner]
target = "profile_a"
start_month = "202503"
end_month = "202601"
dataset_name = "202503_202601"
dry_run = true
collect_only = false

[profiles.profile_a]

[profiles.profile_b]

[profile_groups]
pair_group = ["profile_a", "profile_b"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return project_root


def _shell_wrapper_test_env(
    tmp_path: Path,
    project_root: Path,
    capture_path: Path,
) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_conda = fake_bin / "conda"
    fake_conda.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$ABLATION_TEST_CAPTURE"\n',
        encoding="utf-8",
    )
    fake_conda.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["ALPHAGRAPH_PROJECT_ROOT"] = str(project_root)
    env["ABLATION_TEST_CAPTURE"] = str(capture_path)
    return env
