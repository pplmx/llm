"""Tests for ``llm-quantize`` CLI — currently ``gptq`` subcommand.

The CLI is a thin wrapper around
:func:`llm.quantization.gptq.quantize_model_gptq`, plus mutually-exclusive
calibration input handling and a tokenizer requirement gate. Tests use
``typer.testing.CliRunner`` (same pattern as ``tests/training/test_migrate_ckpt.py``)
so they don't need a real model file or GPU — we exercise the validation
paths and the help output without actually quantizing anything.

Layer 4 of the GPTQ integration plan
(``docs/superpowers/plans/2026-07-22-gptq-integration.md`` § Task 10).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

# Import target — confirmed absent at TDD RED phase, present at GREEN.
from llm.cli.quantize import app


@pytest.fixture
def runner() -> CliRunner:
    """Typer's CliRunner — captures stdout / stderr / exit code.

    Note: ``mix_stderr=False`` is the default in current typer; we keep
    the fixture here so callers always read ``result.stderr`` and
    ``result.stdout`` separately (matches the user-facing contract).
    """
    return CliRunner()


# ---------------------------------------------------------------------------
# Help surface
# ---------------------------------------------------------------------------


def test_cli_help(runner: CliRunner):
    """`llm-quantize gptq --help` exits 0 and lists expected flags."""
    result = runner.invoke(app, ["gptq", "--help"])
    assert result.exit_code == 0, result.output
    assert "--model" in result.output
    assert "--output" in result.output
    assert "--calib-data" in result.output
    assert "--bits" in result.output


def test_cli_root_help_lists_gptq_subcommand(runner: CliRunner):
    """`llm-quantize --help` mentions the `gptq` subcommand."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "gptq" in result.output


# ---------------------------------------------------------------------------
# Required-args gate
# ---------------------------------------------------------------------------


def test_cli_missing_required_args_exits_nonzero(runner: CliRunner):
    """No args → non-zero exit (typer's default missing-required-arg behaviour)."""
    result = runner.invoke(app, ["gptq"])
    assert result.exit_code != 0


def test_cli_missing_model_errors(runner: CliRunner, tmp_path: Path):
    """No --model → error (typer missing-required-arg exit code 2)."""
    result = runner.invoke(app, ["gptq", "--output", str(tmp_path / "out.pt")])
    assert result.exit_code != 0


def test_cli_missing_output_errors(runner: CliRunner, tmp_path: Path):
    """No --output → error."""
    model_path = tmp_path / "model.pt"
    model_path.touch()
    result = runner.invoke(app, ["gptq", "--model", str(model_path)])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_cli_invalid_bits_errors(runner: CliRunner, tmp_path: Path):
    """--bits 16 → error mentioning valid values (must be 4 or 8)."""
    model_path = tmp_path / "model.pt"
    model_path.touch()
    result = runner.invoke(
        app,
        [
            "gptq",
            "--model",
            str(model_path),
            "--output",
            str(tmp_path / "out.pt"),
            "--calib-data",
            str(tmp_path / "calib.txt"),
            "--bits",
            "16",
        ],
    )
    assert result.exit_code != 0
    # Error must reference bits constraint; either by name or by valid value.
    stderr = result.stderr or ""
    stdout = result.output or ""
    assert "bits" in (stderr + stdout).lower()


def test_cli_invalid_group_size_errors(runner: CliRunner, tmp_path: Path):
    """--group-size -2 → error (must be -1 or positive)."""
    model_path = tmp_path / "model.pt"
    model_path.touch()
    result = runner.invoke(
        app,
        [
            "gptq",
            "--model",
            str(model_path),
            "--output",
            str(tmp_path / "out.pt"),
            "--calib-data",
            str(tmp_path / "calib.txt"),
            "--group-size",
            "-2",
        ],
    )
    assert result.exit_code != 0
    stderr = result.stderr or ""
    stdout = result.output or ""
    assert "group" in (stderr + stdout).lower() or "-1" in (stderr + stdout)


# ---------------------------------------------------------------------------
# Calibration input gates
# ---------------------------------------------------------------------------


def test_cli_missing_tokenizer_errors(runner: CliRunner, tmp_path: Path):
    """--calib-data without --tokenizer → error.

    Tokenizing raw text requires an HF tokenizer; without it the CLI must
    refuse to start, not silently produce garbage tokens.
    """
    model_path = tmp_path / "model.pt"
    model_path.touch()
    calib_path = tmp_path / "calib.txt"
    calib_path.write_text("hello world\n")
    result = runner.invoke(
        app,
        [
            "gptq",
            "--model",
            str(model_path),
            "--output",
            str(tmp_path / "out.pt"),
            "--calib-data",
            str(calib_path),
            "--bits",
            "4",
        ],
    )
    assert result.exit_code != 0
    stderr = result.stderr or ""
    assert "tokenizer" in stderr.lower()


def test_cli_calib_data_mutually_exclusive(runner: CliRunner, tmp_path: Path):
    """--calib-data + --calib-data-tokens → error.

    The two inputs describe the same data in two different shapes; the
    CLI refuses both to avoid ambiguous "which one wins" behaviour.
    """
    model_path = tmp_path / "model.pt"
    model_path.touch()
    result = runner.invoke(
        app,
        [
            "gptq",
            "--model",
            str(model_path),
            "--output",
            str(tmp_path / "out.pt"),
            "--calib-data",
            str(tmp_path / "calib.txt"),
            "--calib-data-tokens",
            str(tmp_path / "calib.pt"),
            "--tokenizer",
            str(tmp_path / "tok"),
        ],
    )
    assert result.exit_code != 0
    stderr = result.stderr or ""
    assert "mutually" in stderr.lower() or "exclusive" in stderr.lower()


def test_cli_neither_calib_source_errors(runner: CliRunner, tmp_path: Path):
    """Neither --calib-data nor --calib-data-tokens → error.

    The user must supply calibration data in one of the two supported
    forms; the CLI does not invent defaults (would be silently wrong).
    """
    model_path = tmp_path / "model.pt"
    model_path.touch()
    result = runner.invoke(
        app,
        [
            "gptq",
            "--model",
            str(model_path),
            "--output",
            str(tmp_path / "out.pt"),
            "--bits",
            "4",
        ],
    )
    assert result.exit_code != 0
    stderr = result.stderr or ""
    assert "calib" in stderr.lower()


# ---------------------------------------------------------------------------
# Tokenizer-gated path requires model actually loadable — exercised below
# at the contract level (no real GPU needed because validation rejects
# before any heavy work).
# ---------------------------------------------------------------------------


def test_cli_model_path_must_exist_when_tokenizer_path_resolved(runner: CliRunner, tmp_path: Path):
    """When --calib-data + --tokenizer are supplied, model must exist.

    Tokenizer loading succeeds before the model exists check in the plan;
    we verify the CLI rejects a missing model path with non-zero exit.
    """
    calib_path = tmp_path / "calib.txt"
    calib_path.write_text("hello world\n")
    result = runner.invoke(
        app,
        [
            "gptq",
            "--model",
            str(tmp_path / "missing-model.pt"),
            "--output",
            str(tmp_path / "out.pt"),
            "--calib-data",
            str(calib_path),
            "--tokenizer",
            str(tmp_path / "tok"),
        ],
    )
    assert result.exit_code != 0
    stderr = result.stderr or ""
    assert "model" in stderr.lower() or "not found" in stderr.lower() or "exist" in stderr.lower()
