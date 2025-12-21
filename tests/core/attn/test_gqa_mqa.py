import pytest
import torch

from llm.core.attn.mha import MultiHeadAttention


def test_gqa_initialization_variants():
    """Test different GQA head configurations."""
    hidden_size = 128
    num_heads = 16

    # MQA: 16 query heads, 1 KV head
    mqa = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=1)
    assert mqa.num_kv_heads == 1
    # qkv_proj out_features: (16 + 2*1) * (128//16) = 18 * 8 = 144
    assert mqa.qkv_proj.out_features == 144

    # GQA: 16 query heads, 4 KV heads
    gqa = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=4)
    assert mqa.num_kv_heads == 1
    # qkv_proj out_features: (16 + 2*4) * (128//16) = 24 * 8 = 192
    assert gqa.qkv_proj.out_features == 192


def test_gqa_invalid_config():
    """Test that invalid GQA configs raise errors."""
    hidden_size = 64
    num_heads = 8

    # num_heads must be divisible by num_kv_heads
    with pytest.raises(ValueError, match="num_heads .* must be divisible by num_kv_heads"):
        MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=3)

    # num_kv_heads cannot be greater than num_heads
    with pytest.raises(ValueError, match="num_heads .* must be divisible by num_kv_heads"):
        MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=16)


def test_gqa_numerical_parity_with_mha():
    """Verify GQA with num_kv_heads=num_heads matches standard MHA behavior."""
    hidden_size = 64
    num_heads = 8

    # Standard MHA (implicitly GQA with num_kv_heads=num_heads)
    mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_heads, p=0.0)
    # Explicit GQA with same config
    gqa = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_heads, p=0.0)

    # Copy weights
    gqa.load_state_dict(mha.state_dict())

    x = torch.randn(2, 5, hidden_size)
    with torch.no_grad():
        out_mha = mha(x)
        out_gqa = gqa(x)
        assert torch.allclose(out_mha, out_gqa, atol=1e-6)


def test_gqa_kv_cache_expansion():
    """Deep check on GQA KV Cache dimensions."""
    hidden_size = 64
    num_heads = 8
    num_kv_heads = 2
    mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads)

    x = torch.randn(1, 3, hidden_size)
    _, kv_cache = mha(x, use_cache=True)

    # kv_cache[0] (key) should be [batch, num_kv_heads, seq_len, head_dim]
    assert kv_cache[0].shape == (1, 2, 3, 8)
