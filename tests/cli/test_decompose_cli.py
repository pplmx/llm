"""Tests for the ``llm-decompose`` CLI (low-rank, RIL TASK-225)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from typer.testing import CliRunner

from llm.cli.decompose import app
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


def test_cli_help(runner: CliRunner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "--rank" in out
    assert "--rank-ratio" in out
    assert "--output" in out


def test_cli_requires_exactly_one_rank_knob(runner: CliRunner, tmp_path: Path):
    model = _tiny_decoder(tmp_path / "m.pt")
    out = tmp_path / "o.pt"
    # neither rank nor rank-ratio
    r = runner.invoke(app, ["--model", str(model), "--output", str(out)])
    assert r.exit_code != 0
    assert "exactly one of --rank or --rank-ratio" in strip_ansi(r.output + (r.stderr or ""))
    # both
    r2 = runner.invoke(app, ["--model", str(model), "--output", str(out), "--rank", "4", "--rank-ratio", "0.5"])
    assert r2.exit_code != 0
    assert "exactly one of --rank or --rank-ratio" in strip_ansi(r2.output + (r2.stderr or ""))


def test_cli_invalid_rank_errors(runner: CliRunner, tmp_path: Path):
    model = _tiny_decoder(tmp_path / "m.pt")
    r = runner.invoke(app, ["--model", str(model), "--output", str(tmp_path / "o.pt"), "--rank", "0"])
    assert r.exit_code != 0
    assert "rank" in strip_ansi(r.output + (r.stderr or "")).lower()


def test_cli_refuses_clobber(runner: CliRunner, tmp_path: Path):
    model = _tiny_decoder(tmp_path / "m.pt")
    r = runner.invoke(app, ["--model", str(model), "--output", str(model), "--rank", "4"])
    assert r.exit_code != 0
    assert "must not be the same file" in strip_ansi(r.output + (r.stderr or ""))


def test_cli_end_to_end_decomposes_and_round_trips(runner: CliRunner, tmp_path: Path):
    model_path = _tiny_decoder(tmp_path / "m.pt")
    out_path = tmp_path / "lowrank.pt"

    result = runner.invoke(app, ["--model", str(model_path), "--output", str(out_path), "--rank", "2"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "Compression ratio" in out
    assert "Mean reconstruction error" in out
    assert out_path.exists()

    from llm.quantization.lowrank import LowRankLinear
    from llm.serving.loader import load_training_checkpoint
    from llm.utils.serialization import register_framework_safe_globals

    register_framework_safe_globals()
    lowrank = torch.load(out_path, map_location="cpu", weights_only=True)
    lr_layers = [m for m in lowrank.modules() if isinstance(m, LowRankLinear)]
    assert len(lr_layers) > 0, "no LowRankLinear layers in the CLI output"
    assert sum(1 for m in lowrank.modules() if isinstance(m, nn.Linear)) == 0
    assert all(m.rank == 2 for m in lr_layers)
    with torch.no_grad():
        out_t = lowrank(torch.randint(0, 32, (2, 8)))
    assert torch.isfinite(out_t).all()

    # The low-rank blob serves through the real loader (bare nn.Module blob).
    checkpoint = load_training_checkpoint(out_path)
    assert isinstance(checkpoint.model_obj, torch.nn.Module)
    with torch.no_grad():
        served = checkpoint.model_obj(torch.randint(0, 32, (2, 8)))
    assert torch.isfinite(served).all()
