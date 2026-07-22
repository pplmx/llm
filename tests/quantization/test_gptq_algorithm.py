"""Tests for GPTQ algorithm core (config + Hessian + Cholesky + column loop)."""

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

    with pytest.raises(ValueError, match="blocksize.*must be divisible"):
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
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.bits = 8  # type: ignore[misc]
