"""Tests for attention registry wiring."""

import pytest
import torch

import llm.core.attn.mla  # noqa: F401
from llm.core.registry import ATTENTION_REGISTRY


def test_mla_builder_is_multi_latent_attention():
    assert ATTENTION_REGISTRY.get("mla").__name__ == "MultiLatentAttention"


def test_mla_rejects_kv_cache():
    cls = ATTENTION_REGISTRY.get("mla")
    attn = cls(hidden_size=64, num_heads=4)
    x = torch.randn(1, 4, 64)
    with pytest.raises(ValueError, match="KV cache"):
        attn(x, use_cache=True)
