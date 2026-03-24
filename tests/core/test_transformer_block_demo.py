"""
Transformer Block Demo Tests

Tests TransformerBlock with Pre-LN and Post-LN configurations.
"""

import torch

from llm.core.transformer_block import TransformerBlock


def test_transformer_block_pre_ln():
    """Test TransformerBlock with Pre-LN configuration."""
    device = torch.device("cpu")
    dtype = torch.float32

    hidden_size = 64
    num_heads = 8
    batch_size = 4
    seq_len = 10

    dummy_input = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    pre_ln_block = TransformerBlock(
        hidden_size=hidden_size, num_heads=num_heads, norm_first=True, device=device, dtype=dtype
    )
    pre_ln_block.eval()

    output = pre_ln_block(dummy_input)

    assert output.shape == dummy_input.shape


def test_transformer_block_post_ln():
    """Test TransformerBlock with Post-LN configuration."""
    device = torch.device("cpu")
    dtype = torch.float32

    hidden_size = 64
    num_heads = 8
    batch_size = 4
    seq_len = 10

    dummy_input = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    post_ln_block = TransformerBlock(
        hidden_size=hidden_size, num_heads=num_heads, norm_first=False, device=device, dtype=dtype
    )
    post_ln_block.eval()

    output = post_ln_block(dummy_input)

    assert output.shape == dummy_input.shape


def test_transformer_block_causal():
    """Test TransformerBlock with causal masking."""
    device = torch.device("cpu")
    dtype = torch.float32

    hidden_size = 64
    num_heads = 8
    batch_size = 4
    seq_len = 10

    dummy_input = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    causal_block = TransformerBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        norm_first=True,
        is_causal=True,
        device=device,
        dtype=dtype,
    )
    causal_block.eval()

    # Test with default causality
    output_default = causal_block(dummy_input)
    assert output_default.shape == dummy_input.shape

    # Test with overridden causality
    output_non_causal = causal_block(dummy_input, is_causal=False)
    assert output_non_causal.shape == dummy_input.shape


def test_transformer_block_with_attention_mask():
    """Test TransformerBlock with attention mask."""
    device = torch.device("cpu")
    dtype = torch.float32

    hidden_size = 64
    num_heads = 8
    batch_size = 4
    seq_len = 10

    dummy_input = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    block = TransformerBlock(hidden_size=hidden_size, num_heads=num_heads, norm_first=True, device=device, dtype=dtype)
    block.eval()

    # Create attention mask (mask last token)
    attn_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device, dtype=torch.bool)
    attn_mask[:, :, :, -1] = True

    output = block(dummy_input, attn_mask=attn_mask)

    assert output.shape == dummy_input.shape
