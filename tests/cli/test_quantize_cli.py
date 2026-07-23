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


# ---------------------------------------------------------------------------
# _resolve_target_modules (pure function — no CLI runner needed)
# ---------------------------------------------------------------------------


def test_resolve_target_modules_none_returns_none():
    """None → None (quantize all nn.Linear layers)."""
    from llm.cli.quantize import _resolve_target_modules

    assert _resolve_target_modules(None) is None


def test_resolve_target_modules_single():
    """Single module name → single-element list."""
    from llm.cli.quantize import _resolve_target_modules

    assert _resolve_target_modules("fc1") == ["fc1"]


def test_resolve_target_modules_multiple_with_whitespace():
    """Comma-separated names with whitespace → stripped, empty stripped."""
    from llm.cli.quantize import _resolve_target_modules

    result = _resolve_target_modules(" fc1 , fc2 ,  , fc3 ")
    assert result == ["fc1", "fc2", "fc3"]


def test_resolve_target_modules_empty_string():
    """Empty string → empty list (not None)."""
    from llm.cli.quantize import _resolve_target_modules

    assert _resolve_target_modules("") == []


# ---------------------------------------------------------------------------
# _validate_quant_params: remaining branches
# ---------------------------------------------------------------------------


def test_cli_invalid_percdamp_errors(runner, tmp_path):
    """--percdamp 0 → error (must be in (0, 1))."""
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
            "--calib-data-tokens",
            str(tmp_path / "c.pt"),
            "--percdamp",
            "0",
        ],
    )
    assert result.exit_code != 0


def test_cli_negative_blocksize_errors(runner, tmp_path):
    """--blocksize -1 → error (must be positive)."""
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
            "--calib-data-tokens",
            str(tmp_path / "c.pt"),
            "--blocksize",
            "-1",
        ],
    )
    assert result.exit_code != 0


def test_cli_blocksize_not_divisible_by_group_size(runner, tmp_path):
    """--blocksize 10 --group-size 3 → error (10 % 3 != 0)."""
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
            "--calib-data-tokens",
            str(tmp_path / "c.pt"),
            "--blocksize",
            "10",
            "--group-size",
            "3",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# _validate_model_path: non-file path branch
# ---------------------------------------------------------------------------


def test_cli_model_path_directory_errors(runner, tmp_path):
    """--model pointing at a directory → error (must be a regular file)."""
    dir_path = tmp_path / "model_dir"
    dir_path.mkdir()
    result = runner.invoke(
        app,
        [
            "gptq",
            "--model",
            str(dir_path),
            "--output",
            str(tmp_path / "out.pt"),
            "--calib-data-tokens",
            str(tmp_path / "c.pt"),
        ],
    )
    assert result.exit_code != 0
    stderr = result.stderr or ""
    assert "not a regular file" in stderr.lower() or "not a file" in stderr.lower()
