"""Tests for attention registry wiring."""

from llm.core.registry import ATTENTION_REGISTRY


def test_mla_registered():
    import llm.core.attn.mla  # noqa: F401

    assert "mla" in ATTENTION_REGISTRY
    assert ATTENTION_REGISTRY.get("mla").__name__ == "MultiLatentAttention"
