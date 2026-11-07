"""``llm serve --config`` must not silently fall back to defaults.

``ServingConfig.from_yaml`` deliberately returns the default config for a
missing path (an "absent config means all defaults" convenience used by library
callers, pinned by ``test_missing_file_returns_defaults``). That is the wrong
answer for an EXPLICIT ``--config``: the server started on the defaults —
default model, default host/port — while the operator believed their YAML was
in effect. No warning, no non-zero exit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from llm.serving.cli import serve


@pytest.fixture
def command():
    """Wrap the bare ``serve`` function in a one-command Typer app."""
    app = typer.Typer()
    app.command()(serve)
    return CliRunner(), app


def test_missing_config_path_exits_nonzero_with_an_actionable_message(command, tmp_path: Path):
    runner, app = command
    missing = tmp_path / "does_not_exist.yaml"

    result = runner.invoke(app, ["--config", str(missing)])

    assert result.exit_code != 0, "an explicit --config pointing nowhere must not start the server"
    output = result.output
    assert "not found" in output.lower(), f"the error should say the config is missing, got: {output!r}"
    assert str(missing) in output, "the error should echo the path the user typed"


def test_directory_as_config_path_is_rejected(command, tmp_path: Path):
    """A directory exists, so ``exists()`` alone would let it through."""
    runner, app = command
    a_directory = tmp_path / "config_dir"
    a_directory.mkdir()

    result = runner.invoke(app, ["--config", str(a_directory)])

    assert result.exit_code != 0, "a directory is not a usable config file"


def test_valid_config_file_is_loaded(command, tmp_path: Path, monkeypatch):
    """The guard must not reject a real file; the contents must reach the config."""
    runner, app = command
    config_file = tmp_path / "serve.yaml"
    config_file.write_text("host: 127.0.0.1\nport: 8123\nmodel_path: null\n")

    captured: list = []

    def _fake_main(config) -> None:
        captured.append(config)

    monkeypatch.setattr("llm.serving.cli.main", _fake_main)

    result = runner.invoke(app, ["--config", str(config_file)])

    assert result.exit_code == 0, f"a valid config must be accepted, got: {result.output!r}"
    assert captured, "main() was never called"
    assert captured[0].port == 8123, "the YAML value did not reach the config"
