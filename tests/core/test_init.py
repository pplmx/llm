import pytest

from llm.core.registry import ATTENTION_REGISTRY, MLP_REGISTRY


@pytest.mark.quick
def test_import_llm_registers_components():
    """Verify that importing llm populates core registries."""
    assert "mha" in ATTENTION_REGISTRY.names()
    assert "mla" in ATTENTION_REGISTRY.names()
    assert "flash_attn" in ATTENTION_REGISTRY.names()
    assert "mlp" in MLP_REGISTRY.names()
