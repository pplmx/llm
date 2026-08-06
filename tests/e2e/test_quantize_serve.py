"""End-to-end: ``llm-quantize`` output is directly servable.

Closes the quantization closed loop: a model quantized with the GPTQ CLI
(a bare ``torch.save`` module blob) must be loadable by the serving loader
without losing the quantized layers, and its outputs must match the
quantized artifact it was loaded from.

This is deliberately marked ``e2e`` (opt-in via ``pytest -m e2e``): it runs
the real quantization kernel through the CLI, so it is a few seconds.
"""

from __future__ import annotations

import string

import pytest
import torch
from typer.testing import CliRunner

from llm.cli.quantize import app
from llm.quantization import GPTQQuantizedLinear
from llm.serving.config import ServingConfig
from llm.serving.loader import load_model_and_tokenizer
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from tests.support.devices import DEFAULT_DEVICE


@pytest.mark.e2e
def test_quantize_cli_output_is_servable(tmp_path, tiny_model, tiny_config):
    vocab = tiny_config.model.vocab_size
    model_device = str(next(tiny_model.parameters()).device)

    model_path = tmp_path / "model.pt"
    torch.save(tiny_model, model_path)

    calib = [torch.randint(0, vocab, (2, 16), device=DEFAULT_DEVICE) for _ in range(3)]
    calib_path = tmp_path / "calib.pt"
    torch.save(calib, calib_path)

    output_path = tmp_path / "quantized.pt"
    result = CliRunner().invoke(
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
            "128",
        ],
    )
    assert result.exit_code == 0, result.output

    tokenizer = SimpleCharacterTokenizer(list(string.printable[:vocab]))
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, tokenizer_path)

    config = ServingConfig(
        model_path=str(output_path),
        tokenizer_path=str(tokenizer_path),
        tokenizer_type="simple",
    )
    model, loaded_tokenizer = load_model_and_tokenizer(config)
    model.to(model_device)
    model.eval()

    # The quantized layers survive the CLI → serving round trip.
    assert any(isinstance(m, GPTQQuantizedLinear) for m in model.modules())
    assert loaded_tokenizer.decode(loaded_tokenizer.encode("ab")) == "ab"

    # Output parity with the artifact the server loaded.
    ids = torch.randint(0, vocab, (2, 8), device=model_device)
    reference = torch.load(output_path, map_location=model_device, weights_only=False).eval().to(model_device)
    torch.manual_seed(0)
    expected = reference(ids)
    torch.manual_seed(0)
    actual = model(ids)
    torch.testing.assert_close(actual, expected)
