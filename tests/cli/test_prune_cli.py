"""Tests for the ``llm-prune`` CLI (weight-pruning, RIL TASK-224).

Mirrors ``tests/cli/test_quantize_cli.py`` (CliRunner) plus one real
round-trip: build a tiny ``DecoderModel`` blob, prune it via the CLI, and
load the output back through the same allowlisted `weights_only` path the
serving stack uses, verifying a real sparsity reduction and valid forward.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from typer.testing import CliRunner

from llm.cli.prune import app
from tests.support.ansi import strip_ansi


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _tiny_decoder(path: Path) -> Path:
    from llm.runtime import ModelFactory
    from llm.training.core.config import ModelConfig

    model = ModelFactory.from_config(
        ModelConfig(vocab_size=32, hidden_size=16, num_layers=2, num_heads=2, max_seq_len=24)
    )
    torch.save(model, path)
    return path


def _linear_count(model) -> int:
    return sum(1 for m in model.modules() if isinstance(m, nn.Linear))


def test_cli_help(runner: CliRunner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "--ratio" in out
    assert "--method" in out
    assert "--output" in out


def test_cli_missing_required_args_exits_nonzero(runner: CliRunner):
    result = runner.invoke(app)
    assert result.exit_code != 0


def test_cli_invalid_ratio_errors(runner: CliRunner, tmp_path: Path):
    model = _tiny_decoder(tmp_path / "m.pt")
    result = runner.invoke(app, ["--model", str(model), "--output", str(tmp_path / "o.pt"), "--ratio", "1.0"])
    assert result.exit_code != 0
    assert "ratio" in strip_ansi(result.output + (result.stderr or "")).lower()


def test_cli_invalid_method_errors(runner: CliRunner, tmp_path: Path):
    model = _tiny_decoder(tmp_path / "m.pt")
    result = runner.invoke(app, ["--model", str(model), "--output", str(tmp_path / "o.pt"), "--method", "sensor"])
    assert result.exit_code != 0
    assert "method" in strip_ansi(result.output + (result.stderr or "")).lower()


def test_cli_refuses_clobber(runner: CliRunner, tmp_path: Path):
    model = _tiny_decoder(tmp_path / "m.pt")
    result = runner.invoke(app, ["--model", str(model), "--output", str(model), "--ratio", "0.5"])
    assert result.exit_code != 0
    assert "must not be the same file" in strip_ansi(result.output + (result.stderr or ""))


def test_cli_end_to_end_prunes_and_round_trips(runner: CliRunner, tmp_path: Path):
    model_path = _tiny_decoder(tmp_path / "m.pt")
    out_path = tmp_path / "pruned.pt"

    result = runner.invoke(app, ["--model", str(model_path), "--output", str(out_path), "--ratio", "0.6"])
    assert result.exit_code == 0, result.output
    assert "Achieved sparsity" in strip_ansi(result.output)
    assert out_path.exists()

    # The pruned blob loads through the same allowlisted weights_only path
    # serving uses (llm.quantization.PrunedLinear is allowlisted).
    from llm.quantization.prune import PrunedLinear
    from llm.serving.loader import load_training_checkpoint
    from llm.utils.serialization import register_framework_safe_globals

    register_framework_safe_globals()
    pruned = torch.load(out_path, map_location="cpu", weights_only=True)
    plinear = [m for m in pruned.modules() if isinstance(m, PrunedLinear)]
    assert len(plinear) > 0, "no PrunedLinear layers in the CLI output"
    assert _linear_count(pruned) == 0
    assert all(0.55 <= m.sparsity() <= 0.65 for m in plinear), "sparsity far from requested ratio"
    with torch.no_grad():
        out = pruned(torch.randint(0, 32, (2, 8)))
    assert torch.isfinite(out).all()

    # The pruned blob is served through the real loader (bare nn.Module blob
    # detection) and its forward is valid.
    checkpoint = load_training_checkpoint(out_path)
    assert isinstance(checkpoint.model_obj, torch.nn.Module)
    with torch.no_grad():
        served = checkpoint.model_obj(torch.randint(0, 32, (2, 8)))
    assert torch.isfinite(served).all()
