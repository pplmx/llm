"""End-to-end smoke test for the ``llm-quantize gptq`` CLI.

This complements :mod:`tests.cli.test_quantize_cli` (which exercises
the validation paths via ``CliRunner`` without touching
``quantize_model_gptq``) by exercising the **full CLI happy path**:

1. Save a tiny ``nn.Linear``-only model to disk via ``torch.save``.
2. Save pre-tokenized calibration batches to a ``.pt`` file.
3. Invoke ``llm-quantize gptq --flags`` via ``CliRunner`` (the same
   invocation shape ``pyproject.toml`` ``[project.scripts]`` uses).
4. Verify the output file exists, is a ``torch.save`` blob, and
   contains ``GPTQQuantizedLinear`` modules (i.e. the CLI ran the
   quantization kernel end-to-end — not just argument validation).

The test is marked ``e2e`` so it's opt-in via ``pytest -m e2e`` —
fast enough for daily dev (CPU, ~1s on a tiny model) but excluded
from the ``make test-fast`` lane.

Why this test exists separately from ``test_quantize_cli.py``:

- ``tests/cli/`` tests run validation-only paths and never hit the
  quantization kernel. A regression where ``quantize_model_gptq`` is
  accidentally bypassed (e.g. a refactor that swallows the call) would
  not be caught by validation tests alone — ``--help`` still passes,
  ``--bits 16`` still rejects, but the output ``.pt`` would be
  identical to the input model. This e2e test catches that class of
  regression by asserting the output actually contains
  ``GPTQQuantizedLinear`` modules.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from typer.testing import CliRunner

from llm.cli.quantize import app
from llm.quantization._gptq_layer import GPTQQuantizedLinear


class _TwoLayerMLP(nn.Module):
    """Minimal model with two ``nn.Linear`` layers — smallest valid input."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - not exercised
        return self.fc2(self.act(self.fc1(x)))


@pytest.fixture
def runner() -> CliRunner:
    """CliRunner — same pattern as ``tests/cli/test_quantize_cli.py``."""
    return CliRunner()


@pytest.mark.e2e
def test_cli_quantize_happy_path_replaces_linears(runner: CliRunner, tmp_path):
    """Full CLI invocation: model in → quantized model out.

    Asserts the output ``.pt`` is a real ``GPTQQuantizedLinear``-backed
    model, proving the CLI ran ``quantize_model_gptq`` end-to-end and
    didn't just write the input back unchanged.
    """
    torch.manual_seed(0)
    model = _TwoLayerMLP(hidden=16)
    model_path = tmp_path / "model.pt"
    torch.save(model, model_path)

    # Pre-tokenized calibration: 4 batches of [4, 16] float tensors.
    # Dtype must be float (matches model weights) — torch.randint gives
    # int64 by default and F.linear would reject that on the model
    # forward pass used to capture per-layer inputs.
    calib = [torch.randn(4, 16) for _ in range(4)]
    calib_path = tmp_path / "calib.pt"
    torch.save(calib, calib_path)

    output_path = tmp_path / "out.pt"

    result = runner.invoke(
        app,
        [
            "gptq",
            "--model",
            str(model_path),
            "--output",
            str(output_path),
            "--calib-data-tokens",
            str(calib_path),
            "--bits",
            "4",
            "--group-size",
            "8",
        ],
    )

    assert result.exit_code == 0, (
        f"CLI exited with {result.exit_code}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert output_path.exists(), f"Expected {output_path} to exist"

    # Load the output and verify it actually went through GPTQ.
    quantized = torch.load(output_path, map_location="cpu", weights_only=False)
    linear_count = sum(1 for _ in quantized.modules() if isinstance(_, nn.Linear))
    gptq_count = sum(1 for _ in quantized.modules() if isinstance(_, GPTQQuantizedLinear))

    assert linear_count == 0, f"Expected 0 nn.Linear after quantization, found {linear_count}"
    assert gptq_count == 2, f"Expected 2 GPTQQuantizedLinear (fc1, fc2), found {gptq_count}"


@pytest.mark.e2e
def test_cli_quantize_8bit_path(runner: CliRunner, tmp_path):
    """CLI happy path on the 8-bit width — same shape, different kernel branch.

    The 4-bit and 8-bit code paths in ``_quantize_linear_with_gptq`` are
    near-duplicates (deliberately, to keep packing arithmetic local to
    each width) — this test pins down the 8-bit branch so a refactor
    that breaks one width doesn't silently pass via the other.
    """
    torch.manual_seed(1)
    model = _TwoLayerMLP(hidden=8)
    model_path = tmp_path / "model8.pt"
    torch.save(model, model_path)

    calib = [torch.randn(2, 8) for _ in range(2)]
    calib_path = tmp_path / "calib8.pt"
    torch.save(calib, calib_path)

    output_path = tmp_path / "out8.pt"

    result = runner.invoke(
        app,
        [
            "gptq",
            "--model",
            str(model_path),
            "--output",
            str(output_path),
            "--calib-data-tokens",
            str(calib_path),
            "--bits",
            "8",
            "--group-size",
            "-1",  # per-channel — exercises the -1 branch in the quantization loop
        ],
    )

    assert result.exit_code == 0, (
        f"CLI exited with {result.exit_code}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert output_path.exists()

    quantized = torch.load(output_path, map_location="cpu", weights_only=False)
    gptq_count = sum(1 for _ in quantized.modules() if isinstance(_, GPTQQuantizedLinear))
    assert gptq_count == 2


@pytest.mark.e2e
def test_cli_quantize_output_path_is_created(runner: CliRunner, tmp_path):
    """Output parent dirs are auto-created (deeply nested paths work).

    The CLI's save step calls ``output.parent.mkdir(parents=True, exist_ok=True)``
    so users can write to ``~/models/quantized/int4/run-42/model.pt``
    without pre-creating the directory tree. Pin that contract here.
    """
    model_path = tmp_path / "m.pt"
    torch.save(_TwoLayerMLP(hidden=4), model_path)

    calib_path = tmp_path / "c.pt"
    torch.save([torch.randn(1, 4)], calib_path)

    deep_output = tmp_path / "deep" / "nested" / "int4" / "model.pt"

    result = runner.invoke(
        app,
        [
            "gptq",
            "--model",
            str(model_path),
            "--output",
            str(deep_output),
            "--calib-data-tokens",
            str(calib_path),
            "--bits",
            "4",
            "--group-size",
            "-1",
        ],
    )

    assert result.exit_code == 0, (
        f"CLI exited with {result.exit_code}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert deep_output.exists(), f"Expected {deep_output} to exist after CLI run"
