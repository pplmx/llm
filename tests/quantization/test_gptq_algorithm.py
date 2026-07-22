"""Tests for GPTQ algorithm core (config + Hessian + Cholesky + column loop)."""

from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn as nn

# === Config validation tests ===


def test_gptq_config_default_values():
    """Default config is 4-bit, group_size=128, symmetric."""
    from llm.quantization.gptq import GPTQConfig

    cfg = GPTQConfig()
    assert cfg.bits == 4
    assert cfg.group_size == 128
    assert cfg.sym is True
    assert cfg.percdamp == 0.01
    assert cfg.blocksize == 128
    assert cfg.act_order is False
    assert cfg.static_groups is False


def test_gptq_config_rejects_invalid_bits():
    """bits must be 4 or 8."""
    from llm.quantization.gptq import GPTQConfig

    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        GPTQConfig(bits=16)


def test_gptq_config_rejects_negative_group_size():
    """group_size must be -1 (per-channel) or positive."""
    from llm.quantization.gptq import GPTQConfig

    with pytest.raises(ValueError, match="group_size must be -1"):
        GPTQConfig(group_size=-128)


def test_gptq_config_rejects_invalid_percdamp():
    """percdamp must be in (0, 1)."""
    from llm.quantization.gptq import GPTQConfig

    with pytest.raises(ValueError, match="percdamp must be in"):
        GPTQConfig(percdamp=0.0)


def test_gptq_config_rejects_nonpositive_blocksize():
    """blocksize must be positive."""
    from llm.quantization.gptq import GPTQConfig

    with pytest.raises(ValueError, match="blocksize must be positive"):
        GPTQConfig(blocksize=0)


def test_gptq_config_rejects_blocksize_not_divisible_by_group_size():
    """When group_size > 0, blocksize must be divisible by group_size."""
    from llm.quantization.gptq import GPTQConfig

    with pytest.raises(ValueError, match=r"blocksize.*must be divisible"):
        GPTQConfig(group_size=128, blocksize=100)


def test_gptq_config_per_channel_skips_divisibility_check():
    """group_size=-1 (per-channel) skips blocksize divisibility check."""
    from llm.quantization.gptq import GPTQConfig

    cfg = GPTQConfig(group_size=-1, blocksize=100)  # should not raise
    assert cfg.group_size == -1
    assert cfg.blocksize == 100


def test_gptq_config_is_frozen():
    """Config is frozen (immutable)."""
    from llm.quantization.gptq import GPTQConfig

    cfg = GPTQConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.bits = 4  # type: ignore[misc]


# === GPTQQuantizer Hessian accumulation tests ===


def test_quantizer_initializes_with_zero_hessian():
    """Fresh GPTQQuantizer has H == 0 and no samples accumulated."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    layer = nn.Linear(8, 4, bias=False)
    cfg = GPTQConfig()
    q = GPTQQuantizer(layer, cfg)

    assert q.n_samples == 0
    assert torch.allclose(q.H, torch.zeros_like(q.H))


def test_add_batch_handles_2d_input():
    """add_batch accepts [batch, in_features] tensor."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    layer = nn.Linear(8, 4, bias=False)
    cfg = GPTQConfig()
    q = GPTQQuantizer(layer, cfg)

    x = torch.randn(16, 8)
    q.add_batch(x)

    assert q.n_samples == 16
    # H should be non-zero after add_batch
    assert q.H.abs().sum() > 0


def test_add_batch_handles_3d_input():
    """add_batch accepts [batch, seq, in_features] tensor (flattens batch+seq)."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    layer = nn.Linear(8, 4, bias=False)
    cfg = GPTQConfig()
    q = GPTQQuantizer(layer, cfg)

    x = torch.randn(4, 5, 8)  # 4*5 = 20 samples
    q.add_batch(x)

    assert q.n_samples == 20


def test_add_batch_accumulates_hessian_correctly():
    """Multiple add_batches accumulate H = 2/N · Σ X^T X."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    in_features = 6
    layer = nn.Linear(in_features, 4, bias=False)
    cfg = GPTQConfig()
    q = GPTQQuantizer(layer, cfg)

    # Two batches
    x1 = torch.randn(10, in_features)
    x2 = torch.randn(8, in_features)
    q.add_batch(x1)
    q.add_batch(x2)

    # Expected: H = 2/18 * (x1.T @ x1 + x2.T @ x2)
    expected = (2.0 / 18) * (x1.t() @ x1 + x2.t() @ x2)

    assert torch.allclose(q.H, expected, atol=1e-5)


def test_add_batch_matches_one_shot():
    """Multi-batch accumulate equals single concatenated add_batch."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    in_features = 6
    layer = nn.Linear(in_features, 4, bias=False)
    cfg = GPTQConfig()

    # Same data processed as two chunks vs. one concatenated tensor.
    torch.manual_seed(0)
    chunk_a = torch.randn(5, in_features)
    chunk_b = torch.randn(7, in_features)
    full = torch.cat([chunk_a, chunk_b], dim=0)

    # Multi-batch
    q1 = GPTQQuantizer(layer, cfg)
    q1.add_batch(chunk_a)
    q1.add_batch(chunk_b)

    # Single-shot
    q2 = GPTQQuantizer(layer, cfg)
    q2.add_batch(full)

    assert torch.allclose(q1.H, q2.H, atol=1e-5)
    assert q1.n_samples == q2.n_samples == 12


# === GPTQ algorithm correctness tests ===


def test_gptq_lower_error_than_rtn_baseline():
    """GPTQ MSE on a small layer must be lower than naive round-to-nearest.

    This is the core promise of GPTQ — Hessian-aware quantization beats
    per-column rounding. Tested on a small dense layer with calibration data.
    """
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    torch.manual_seed(42)
    in_f, out_f = 16, 16
    layer = nn.Linear(in_f, out_f, bias=False)
    # Inject meaningful structure (not random init)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(out_f, in_f) * 0.5)

    # Generate calibration data with structure
    calib = torch.randn(64, in_f)

    # GPTQ
    q = GPTQQuantizer(layer, GPTQConfig(bits=4, group_size=-1))
    q.add_batch(calib)
    w_q, scales, _zeros = q.quantize()

    # Reconstruct W_q in fp32 for comparison
    # scales shape: [out_f, 1] for group_size=-1
    w_recon = w_q.float() * scales.float()
    mse_gptq = ((layer.weight - w_recon) ** 2).mean().item()

    # RTN baseline: per-channel symmetric 4-bit
    abs_max = layer.weight.abs().max(dim=1, keepdim=True)[0]
    qmax = 2 ** (4 - 1) - 1  # 7
    scale_rtn = abs_max / qmax
    w_rtn = (layer.weight / scale_rtn).round().clamp(-8, 7) * scale_rtn
    mse_rtn = ((layer.weight - w_rtn) ** 2).mean().item()

    assert mse_gptq < mse_rtn, f"GPTQ MSE {mse_gptq:.6f} should beat RTN MSE {mse_rtn:.6f}"


def test_quantize_handles_zero_calibration_gracefully():
    """Zero calibration data → actionable error mentioning percdamp."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    layer = nn.Linear(8, 4, bias=False)
    q = GPTQQuantizer(layer, GPTQConfig(percdamp=0.01))

    with pytest.raises(RuntimeError, match="percdamp"):
        q.quantize()


def test_quantize_handles_singular_hessian_with_higher_damp():
    """Rank-deficient Hessian succeeds with sufficient damping."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    torch.manual_seed(0)
    in_f, out_f = 8, 4
    layer = nn.Linear(in_f, out_f, bias=False)
    # Calibration data with rank < in_f (constant feature)
    calib = torch.zeros(16, in_f)
    calib[:, 0] = torch.linspace(-1, 1, 16)  # only first dim varies

    # 0.01 damp fails, 0.5 damp succeeds
    q1 = GPTQQuantizer(layer, GPTQConfig(percdamp=0.01))
    q1.add_batch(calib)
    with pytest.raises(RuntimeError):
        q1.quantize()

    q2 = GPTQQuantizer(layer, GPTQConfig(percdamp=0.5))
    q2.add_batch(calib)
    w_q, _scales, _zeros = q2.quantize()
    assert w_q.shape == (out_f, in_f)


def test_quantize_returns_correct_shapes():
    """quantize() returns W_q [out, in], scales [out, in/group_size] (or [out,1] for per-channel)."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    in_f, out_f = 32, 16
    layer = nn.Linear(in_f, out_f, bias=False)
    calib = torch.randn(32, in_f)

    # group_size=8 → 4 groups per row
    q = GPTQQuantizer(layer, GPTQConfig(group_size=8))
    q.add_batch(calib)
    w_q, scales, _zeros = q.quantize()
    assert w_q.shape == (out_f, in_f)
    assert scales.shape == (out_f, in_f // 8)

    # group_size=-1 → 1 group per row (per-channel)
    q2 = GPTQQuantizer(layer, GPTQConfig(group_size=-1))
    q2.add_batch(calib)
    w_q2, scales2, _zeros2 = q2.quantize()
    assert w_q2.shape == (out_f, in_f)
    assert scales2.shape == (out_f, 1)
