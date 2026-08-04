from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tomllib

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALPHAGRAPH_DIR = PROJECT_ROOT / "alphagraph"
LOADER_PATH = ALPHAGRAPH_DIR / "load_conda_env.sh"
SHELL_ENTRYPOINTS = sorted(ALPHAGRAPH_DIR.glob("*.sh")) + sorted(
    (PROJECT_ROOT / "scripts").glob("*.sh")
)


def _load_env_from_config(tmp_path: Path, config_text: str, ambient: str = "ambient"):
    config_path = tmp_path / "pipeline_config.toml"
    config_path.write_text(config_text, encoding="utf-8")
    command = (
        "set -euo pipefail; "
        f'source "{LOADER_PATH}"; '
        f'ENV_NAME="{ambient}"; '
        f'load_conda_env "{config_path}"; '
        "printf '%s' \"$ENV_NAME\""
    )
    return subprocess.run(
        ["bash", "-c", command],
        text=True,
        capture_output=True,
        check=False,
    )


def test_pipeline_config_is_the_single_conda_environment_source():
    config_path = ALPHAGRAPH_DIR / "pipeline_config.toml"
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    assert config["defaults"]["env_name"] == "rGNN"
    config_text = config_path.read_text(encoding="utf-8")
    assert len(re.findall(r"^[ \t]*env_name[ \t]*=", config_text, re.MULTILINE)) == 1


def test_shell_entrypoints_use_shared_environment_loader():
    for script_path in SHELL_ENTRYPOINTS:
        script = script_path.read_text(encoding="utf-8")
        if not re.search(r"\bpython(?:3)?\b", script):
            continue
        assert "load_conda_env.sh" in script, script_path
        assert "load_conda_env " in script, script_path
        assert "python3" not in script, script_path
        assert not re.search(r"\b(?:learn|rGNN)\b", script), script_path


def test_loader_reads_defaults_and_overrides_ambient_environment(tmp_path: Path):
    result = _load_env_from_config(
        tmp_path,
        """
[defaults]
env_name = "rGNN" # server environment

[mode_defaults.gnn]
epochs = 20
""".strip(),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "rGNN"


@pytest.mark.parametrize(
    "config_text",
    [
        "[defaults]\nartifact_namespace = \"alphagraph\"",
        "[defaults]\nenv_name = \"bad env\"",
        "[other]\nenv_name = \"rGNN\"",
    ],
)
def test_loader_rejects_missing_malformed_or_out_of_section_values(
    tmp_path: Path,
    config_text: str,
):
    result = _load_env_from_config(tmp_path, config_text)

    assert result.returncode != 0
    assert "env_name" in result.stderr
