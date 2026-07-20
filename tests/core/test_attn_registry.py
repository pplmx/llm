"""Tests for attention registry wiring."""

import torch

import llm.core.attn.mla  # noqa: F401
from llm.core.registry import (
    ATTENTION_KV_CACHE_CAPABILITY,
    ATTENTION_REGISTRY,
    attention_supports_kv_cache,
)


def test_mla_builder_is_multi_latent_attention():
    assert ATTENTION_REGISTRY.get("mla").__name__ == "MultiLatentAttention"


def test_mla_kv_cache_capability_declared():
    """MLA declares its KV-cache capability at import time (Tier 3 #31).

    Regression for Finding E (audit 2026-07-12): the capability flag must
    match what the forward path actually supports — otherwise
    ``ModelConfig.check_consistency`` would accept configurations that
    fail at runtime.
    """
    assert attention_supports_kv_cache("mla") is True
    assert ATTENTION_KV_CACHE_CAPABILITY["mla"] is True


def test_mla_forward_with_use_cache_returns_tuple():
    """``use_cache=True`` returns ``(output, None)`` — MLA's contract.

    Unlike MHA (which surfaces the cached K, V), the placeholder MLA
    consumes the cached K, V internally and returns ``None`` as the
    second element. The paged path returns the bare output tensor
    instead — see the MLA test suite for both branches.
    """
    cls = ATTENTION_REGISTRY.get("mla")
    attn = cls(hidden_size=64, num_heads=4, p=0.0, include_norm_residual=False).eval()
    x = torch.randn(1, 4, 64)
    with torch.no_grad():
        out = attn(x, use_cache=True)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[1] is None  # MLA does not surface K, V externally.
