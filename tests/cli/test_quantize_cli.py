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
from tests.support.ansi import strip_ansi


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
    out = strip_ansi(result.output)
    assert "--model" in out
    assert "--output" in out
    assert "--calib-data" in out
    assert "--bits" in out


def test_cli_root_help_lists_gptq_subcommand(runner: CliRunner):
    """`llm-quantize --help` mentions the `gptq` subcommand."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "gptq" in strip_ansi(result.output)


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
    stderr = strip_ansi(result.stderr or "")
    stdout = strip_ansi(result.output or "")
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
    stderr = strip_ansi(result.stderr or "")
    stdout = strip_ansi(result.output or "")
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
    stderr = strip_ansi(result.stderr or "")
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
    stderr = strip_ansi(result.stderr or "")
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
    stderr = strip_ansi(result.stderr or "")
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
    stderr = strip_ansi(result.stderr or "")
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
    """Empty / whitespace-only input → None (no filter = all nn.Linear layers).

    The old ``[]`` (empty list) made the CLI echo ``all`` in the summary while
    actually matching ZERO layers — ``quantize_model_gptq`` then failed with
    an exit-2 runtime error. ``None`` keeps the summary and the behaviour in
    agreement.
    """
    from llm.cli.quantize import _resolve_target_modules

    assert _resolve_target_modules("") is None
    assert _resolve_target_modules("  , , ") is None


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
    stderr = strip_ansi(result.stderr or "")
    assert "not a regular file" in stderr.lower() or "not a file" in stderr.lower()


# ---------------------------------------------------------------------------
# _load_calibration_batches: calib_data_tokens shape-handling branches
# ---------------------------------------------------------------------------


def test_load_calibration_batches_single_tensor(tmp_path: Path):
    """A single .pt tensor is wrapped in a list (uniform iterable contract)."""
    import torch

    from llm.cli.quantize import _load_calibration_batches

    calib = tmp_path / "calib.pt"
    torch.save(torch.tensor([[1, 2, 3]]), calib)

    result = _load_calibration_batches(None, calib, None)
    assert isinstance(result, list)
    assert len(result) == 1
    assert torch.equal(result[0], torch.tensor([[1, 2, 3]]))


def test_load_calibration_batches_list_of_tensors(tmp_path: Path):
    """A .pt list of tensors is returned as-is (already iterable)."""
    import torch

    from llm.cli.quantize import _load_calibration_batches

    calib = tmp_path / "calib.pt"
    torch.save([torch.tensor([[1, 2]]), torch.tensor([[3, 4]])], calib)

    result = _load_calibration_batches(None, calib, None)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (1, 2)
    assert result[1].shape == (1, 2)


def test_load_calibration_batches_rejects_unknown_shape(tmp_path: Path):
    """A calibration payload that is neither a tensor nor a list of tensors is
    rejected at load time with a clear error.

    The old passthrough handed a dict straight to the quantizer, which
    iterated the dict *keys* and failed downstream with a confusing deep
    error during Hessian capture.
    """
    import torch

    from llm.cli.quantize import _load_calibration_batches

    calib = tmp_path / "calib.pt"
    torch.save({"unexpected": "dict"}, calib)

    with pytest.raises(RuntimeError, match="tensor"):
        _load_calibration_batches(None, calib, None)


def test_load_calibration_batches_rejects_non_tensor_list(tmp_path: Path):
    """A list with non-tensor entries is rejected at load time too."""
    import torch

    from llm.cli.quantize import _load_calibration_batches

    calib = tmp_path / "calib.pt"
    torch.save([torch.tensor([[1, 2]]), "oops"], calib)

    with pytest.raises(RuntimeError, match="non-tensor"):
        _load_calibration_batches(None, calib, None)


def test_cli_asym_rejected_as_argument_error(runner: CliRunner, tmp_path: Path):
    """``--asym`` is not implemented anywhere in GPTQ — it must be rejected at
    argument validation (exit 1) with a clear message, not sail through the
    checks and fail as an exit-2 ``quantization failed`` deep in the Hessian
    loop."""
    import torch

    model_path = tmp_path / "model.pt"
    model_path.touch()
    calib = tmp_path / "calib.pt"
    torch.save(torch.tensor([[1, 2, 3]], dtype=torch.float32), calib)

    result = runner.invoke(
        app,
        [
            "gptq",
            "--model",
            str(model_path),
            "--output",
            str(tmp_path / "out.pt"),
            "--calib-data-tokens",
            str(calib),
            "--asym",
        ],
    )
    assert result.exit_code == 1, f"got {result.exit_code}: {result.stdout!r} {result.stderr!r}"
    combined = strip_ansi(result.stderr or "") + strip_ansi(result.stdout or "")
    assert "asymmetric" in combined.lower()
    # Must be an argument-validation SystemExit, not a raw exception.
    assert isinstance(result.exception, SystemExit), f"raw exception escaped: {result.exception!r}"


def test_fp8_default_activation_is_dynamic_and_needs_no_calib(runner: CliRunner, tmp_path: Path):
    """Regression: ``static`` being the default made the simplest documented
    ``llm-quantize fp8 --model ... --output ...`` call exit 1 demanding
    calibration. The default must be ``dynamic`` (per-forward scaling, no
    calibration)."""
    import torch

    model_path = tmp_path / "model.pt"
    # A loadable-but-not-a-model payload: proves validation + load succeeded
    # and the failure came from the later quantization stage, not the calib gate.
    torch.save({"not_a_model": 1}, model_path)

    result = runner.invoke(
        app,
        ["fp8", "--model", str(model_path), "--output", str(tmp_path / "out.pt")],
    )
    assert result.exit_code == 2, f"got {result.exit_code}: {result.stdout!r} {result.stderr!r}"
    combined = strip_ansi(result.stderr or "") + strip_ansi(result.stdout or "")
    assert "must supply calibration" not in combined


def test_fp8_static_without_calib_exits_1_with_hint(runner: CliRunner, tmp_path: Path):
    """``--activation static`` with no calibration source keeps a clean exit-1
    error, and the message points at the escape hatch (``--activation
    dynamic``)."""
    model_path = tmp_path / "model.pt"
    model_path.touch()

    result = runner.invoke(
        app,
        [
            "fp8",
            "--model",
            str(model_path),
            "--output",
            str(tmp_path / "out.pt"),
            "--activation",
            "static",
        ],
    )
    assert result.exit_code == 1
    combined = strip_ansi(result.stderr or "") + strip_ansi(result.stdout or "")
    assert "--activation dynamic" in combined


def test_load_calibration_tokens_refuses_malicious_pickle(tmp_path: Path):
    """Regression (RIL ISS-211): a pre-tokenized calibration file must NOT
    execute arbitrary ``__reduce__`` code.

    The old ``weights_only=False`` would run ``os.system`` from a crafted
    third-party ``--calib-data-tokens`` file; the hardened load refuses it
    before any code runs (same posture as the serving loader, ISS-170).
    """
    import os
    import pickle

    marker = tmp_path / "pwned"
    evil = tmp_path / "evil.pt"

    class _Exploit:
        def __reduce__(self):  # pragma: no cover - must never run
            return (os.system, (f"touch {marker}",))

    with evil.open("wb") as fh:
        pickle.dump(_Exploit(), fh, protocol=2)

    from llm.cli.quantize import _load_calibration_batches

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        _load_calibration_batches(None, evil, None)
    assert not marker.exists(), "malicious calibration pickle executed code"


# ---------------------------------------------------------------------------
# ISS-161: clobber guard + atomic save
# ---------------------------------------------------------------------------


def test_cli_rejects_output_equal_to_model(runner, tmp_path):
    """Regression (RIL ISS-161): ``--output == --model`` must exit 1 before
    quantization, not overwrite the source checkpoint with the quantized
    model."""
    model_path = tmp_path / "model.pt"
    model_path.touch()
    result = runner.invoke(
        app,
        [
            "gptq",
            "--model",
            str(model_path),
            "--output",
            str(model_path),  # same file!
            "--calib-data-tokens",
            str(tmp_path / "c.pt"),
        ],
    )
    assert result.exit_code == 1
    stderr = strip_ansi(result.stderr or "")
    assert "same file" in stderr.lower()


def test_cli_rejects_output_aliasing_model_via_resolved_path(runner, tmp_path):
    """Regression (RIL ISS-161): `--output ./model.pt` with `--model
    model.pt` refers to the same file — the resolve()-based guard must
    catch the alias, not just the byte-identical string."""
    model_path = tmp_path / "model.pt"
    model_path.touch()
    result = runner.invoke(
        app,
        [
            "gptq",
            "--model",
            str(model_path),
            "--output",
            str(tmp_path) + "/./model.pt",  # same file via a different spelling
            "--calib-data-tokens",
            str(tmp_path / "c.pt"),
        ],
    )
    assert result.exit_code == 1
    stderr = strip_ansi(result.stderr or "")
    assert "same file" in stderr.lower()


def test_atomic_save_leaves_no_tmp_and_writes_loadable_output(tmp_path):
    """Regression (RIL ISS-161): the quantization write goes through a temp
    file + atomic rename. A successful save must leave no ``.tmp`` sibling
    and the output must load as the saved object."""
    import torch

    from llm.cli.quantize import _atomic_save_quantized

    payload = {"kind": "quantized", "value": 42, "tensor": torch.ones(2, 3)}
    output = tmp_path / "quant.pt"
    _atomic_save_quantized(payload, output)

    assert output.exists()
    assert not output.with_suffix(output.suffix + ".tmp").exists()
    loaded = torch.load(output, map_location="cpu", weights_only=False)
    assert loaded["value"] == 42
    assert torch.equal(loaded["tensor"], torch.ones(2, 3))
