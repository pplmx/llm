"""FP8 (E4M3/E5M2) PTQ tests — layer, model e2e, size accounting and CLI.

Verification strategy: FP8 is a post-training quantization, so we assert
CLOSE (not bit-exact) reconstruction — the quantized forward must be a
faithful (few-percent) approximation of the fp32 reference, stay finite for
both static and dynamic activation scaling, and genuinely shrink the stored
checkpoint (real float8 weights, 1 byte each).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from llm.quantization._fp8_layer import FP8_MAX, Fp8QuantizedLinear, quantize_fp8_linear
from llm.quantization.fp8 import Fp8Config, quantize_model_fp8
from llm.quantization.ptq import compute_model_size


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)).item())


# ---------------------------------------------------------------------------
# Layer / format
# ---------------------------------------------------------------------------


def test_fp8_weight_is_real_float8_storage():
    """The layer must store 1-byte float8 weights, not fp32 pretending."""
    layer = nn.Linear(16, 8)
    q = quantize_fp8_linear(layer, activation_scale=torch.tensor(1.0))
    assert isinstance(q, Fp8QuantizedLinear)
    assert q.weight_fp8.dtype == torch.float8_e4m3fn
    assert q.weight_fp8.element_size() == 1
    assert q.weight_fp8.shape == layer.weight.shape


def test_fp8_layer_forward_is_close_to_fp32_reference():
    layer = nn.Linear(32, 16)
    torch.manual_seed(3)
    x = torch.randn(4, 32)
    q = quantize_fp8_linear(layer, activation_scale=torch.tensor(1.0))
    with torch.no_grad():
        y_ref = layer(x)
        y_q = q(x)
    assert torch.isfinite(y_q).all()
    # E4M3 per-tensor keeps the output highly correlated with the fp32 model.
    assert _cosine(y_q, y_ref) > 0.999, (y_q - y_ref).abs().max().item()
    # Relative error stays bounded (per-tensor E4M3 ~ few percent).
    rel = float((y_q - y_ref).abs().max() / (y_ref.abs().max() + 1e-8))
    assert rel < 0.05, rel


def test_fp8_e5m2_variant_roundtrips():
    layer = nn.Linear(8, 4)
    q = quantize_fp8_linear(layer, dtype_name="e5m2", activation_scale=torch.tensor(1.0))
    assert q.weight_fp8.dtype == torch.float8_e5m2
    x = torch.randn(2, 8)
    out = q(x)
    assert torch.isfinite(out).all()


def test_fp8_rejects_unknown_dtype():
    layer = nn.Linear(8, 4)
    with pytest.raises(ValueError, match="Unsupported FP8 dtype"):
        quantize_fp8_linear(layer, dtype_name="fp4")


def test_fp8_dequant_stays_within_format_range():
    """The fp8-cast weight must not silently exceed the format max."""
    layer = nn.Linear(16, 8)
    # A weight with a huge outlier maps the whole tensor onto ±448 exactly.
    with torch.no_grad():
        layer.weight[0, 0] = 1e3
    q = quantize_fp8_linear(layer, activation_scale=torch.tensor(1.0))
    # The STORED fp8 values must sit inside the format range (saturating cast,
    # no silent overflow); the dequantized weights restore the original
    # magnitude, so only the stored grid is range-bounded.
    assert float(q.weight_fp8.float().abs().max()) <= FP8_MAX["e4m3"]
    assert float(q.weight_fp8.float().abs().max()) >= FP8_MAX["e4m3"] * 0.99  # outlier saturates near max


# ---------------------------------------------------------------------------
# Model e2e
# ---------------------------------------------------------------------------


def _tiny_model(seed: int = 7) -> nn.Module:
    from llm.models.decoder import DecoderModel

    torch.manual_seed(seed)
    return DecoderModel(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        max_seq_len=24,
        intermediate_size=64,
        attn_dropout_p=0.0,
        mlp_dropout_p=0.0,
        embedding_dropout_p=0.0,
        qkv_bias=True,
        mlp_bias=True,
        lm_head_bias=True,
    )


def test_quantize_model_fp8_replaces_linears():
    model = _tiny_model()
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    calib = [torch.randint(0, 64, (2, 12))]
    model.eval()
    quantized = quantize_model_fp8(model, iter(calib), Fp8Config())
    replaced = sum(1 for m in quantized.modules() if isinstance(m, Fp8QuantizedLinear))
    assert replaced == n_linear
    # The fp8 model runs and stays finite.
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        logits = quantized(x)
    assert torch.isfinite(logits).all()
    assert logits.shape == (2, 12, 64)


def test_quantize_model_fp8_logits_correlate_with_fp32_baseline():
    calib = [torch.randint(0, 64, (4, 12)) for _ in range(3)]
    base = _tiny_model().eval()
    with torch.no_grad():
        ref = base(torch.randint(0, 64, (2, 12)))
    quantized = quantize_model_fp8(_tiny_model().eval(), iter(calib), Fp8Config())
    with torch.no_grad():
        q = quantized(torch.randint(0, 64, (2, 12)))
    # Same architecture/init; FP8 stays a faithful approximation (not exact —
    # quantization inherently perturbs logits).
    assert _cosine(q.float(), ref.float()) > 0.99, (q.float() - ref.float()).abs().max().item()


def test_quantize_model_fp8_static_requires_calibration():
    with pytest.raises(ValueError, match="calib_iter"):
        quantize_model_fp8(_tiny_model(), None, Fp8Config(activation="static"))


def test_quantize_model_fp8_dynamic_needs_no_calibration():
    model = _tiny_model().eval()
    quantized = quantize_model_fp8(model, None, Fp8Config(activation="dynamic"))
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        logits = quantized(x)
    assert torch.isfinite(logits).all()


def test_quantize_model_fp8_rejects_double_quantization():
    model = _tiny_model()
    quantize_model_fp8(model, iter([torch.randint(0, 64, (2, 12))]), Fp8Config())
    with pytest.raises(ValueError, match="already FP8"):
        quantize_model_fp8(model, iter([torch.randint(0, 64, (2, 12))]), Fp8Config())


# ---------------------------------------------------------------------------
# Size accounting
# ---------------------------------------------------------------------------


def test_compute_model_size_counts_fp8_quantized_layers():
    model = _tiny_model()
    fp32_stats = compute_model_size(model)
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    quantize_model_fp8(model, iter([torch.randint(0, 64, (2, 12))]), Fp8Config())
    stats = compute_model_size(model)
    assert stats["quantized_layers"] == n_linear
    assert stats["total_params"] == fp32_stats["total_params"]  # same weight count
    # FP8 stores weights at 1 byte each (fp32 = 4) + small fp32 scales/bias.
    assert stats["total_bytes"] < fp32_stats["total_bytes"]
    assert stats["total_bytes"] < 2 * fp32_stats["total_params"]  # <2 bytes/param incl scale+bias


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    from typer.testing import CliRunner

    return CliRunner()


def _save_tiny_model(path: Path) -> None:
    torch.save(nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4)), path)


def test_cli_root_help_lists_fp8_subcommand(runner):
    from llm.cli.quantize import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "fp8" in result.stdout


def test_cli_fp8_rejects_bad_weight_dtype(runner, tmp_path: Path):
    from llm.cli.quantize import app

    model = tmp_path / "m.pt"
    _save_tiny_model(model)
    result = runner.invoke(
        app,
        ["fp8", "--model", str(model), "--output", str(tmp_path / "o.pt"), "--weight-dtype", "xfp8"],
    )
    assert result.exit_code == 1
    assert "weight-dtype" in result.stderr


def test_cli_fp8_dynamic_needs_no_calib_and_saves(runner, tmp_path: Path):
    from llm.cli.quantize import app

    model = tmp_path / "m.pt"
    out = tmp_path / "o.pt"
    _save_tiny_model(model)
    result = runner.invoke(
        app,
        ["fp8", "--model", str(model), "--output", str(out), "--activation", "dynamic", "--per-tensor"],
    )
    assert result.exit_code == 0, result.stderr
    assert out.exists()
    loaded = torch.load(out, map_location="cpu")
    quantized = sum(1 for m in loaded.modules() if isinstance(m, Fp8QuantizedLinear))
    assert quantized == 2


def test_cli_fp8_static_requires_calib_source(runner, tmp_path: Path):
    from llm.cli.quantize import app

    model = tmp_path / "m.pt"
    _save_tiny_model(model)
    result = runner.invoke(app, ["fp8", "--model", str(model), "--output", str(tmp_path / "o.pt")])
    assert result.exit_code == 1
    assert "calibration" in result.stderr
