"""llm-train CLI flag-contract tests (RIL ISS-411).

``--compile`` / ``--amp`` must be *overrides*: when the flag is omitted the
YAML config stays authoritative. A ``True`` typer default previously
unconditionally re-enabled ``use_compile`` / ``use_amp`` on every run whose
config deliberately disabled them.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from llm.training.train import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_cfg(tmp_path, use_compile: bool, use_amp: bool):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"optimization:\n  use_compile: {str(use_compile).lower()}\n  use_amp: {str(use_amp).lower()}\n")
    return cfg


def _run_and_capture(runner, monkeypatch, args):
    """Invoke ``main`` with ``train_worker`` faked out, returning the
    ``config`` the training loop would have received."""
    seen: dict[str, bool | None] = {}

    def fake_train_worker(_rank, _world_size, config, _task, *_args, **_kwargs):
        seen["use_compile"] = config.optimization.use_compile
        seen["use_amp"] = config.optimization.use_amp

    monkeypatch.setattr("llm.training.train.train_worker", fake_train_worker)
    # The test host may advertise GPUS_PER_NODE > 1 (an env-driven
    # DistributedConfig), which routes main through mp.spawn; force the
    # single-process path so the fake worker is invoked directly.
    monkeypatch.setattr("llm.training.train.DistributedManager.get_world_size", lambda self: 1)
    result = runner.invoke(app, args)
    return result, seen


@pytest.mark.quick
def test_train_cli_omitted_compile_amp_keep_yaml(runner, monkeypatch, tmp_path):
    """A YAML config that disabled compile/AMP must NOT be silently flipped
    back on when the user passes neither flag (RIL ISS-411)."""
    cfg = _write_cfg(tmp_path, use_compile=False, use_amp=False)
    result, seen = _run_and_capture(runner, monkeypatch, ["--task", "lm", "--config-path", str(cfg)])
    assert result.exit_code == 0, result.stdout
    assert seen["use_compile"] is False, "config's use_compile=False must survive an omitted --compile"
    assert seen["use_amp"] is False, "config's use_amp=False must survive an omitted --amp"


@pytest.mark.quick
def test_train_cli_explicit_flags_override_yaml(runner, monkeypatch, tmp_path):
    """An EXPLICIT ``--compile`` / ``--no-amp`` still overrides the YAML
    value (the flags are overrides, not defaults)."""
    cfg = _write_cfg(tmp_path, use_compile=False, use_amp=True)
    result, seen = _run_and_capture(
        runner, monkeypatch, ["--task", "lm", "--config-path", str(cfg), "--compile", "--no-amp"]
    )
    assert result.exit_code == 0, result.stdout
    assert seen["use_compile"] is True
    assert seen["use_amp"] is False
