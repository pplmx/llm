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
