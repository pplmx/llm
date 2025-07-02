import math

import pytest
import torch
import torch.nn.functional as F

from llm.core.attn.dot_product_attn import scaled_dot_product_attention


@pytest.fixture
def sample_qkv_tensors():
    """Fixture to provide sample Q, K, V tensors."""
    batch_size = 2
    num_heads = 4
    seq_len_q = 5
    seq_len_k = 7  # Key/Value sequence length can differ from query
    head_dim = 8

    query = torch.randn(batch_size, num_heads, seq_len_q, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len_k, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len_k, head_dim)
    return query, key, value


def test_scaled_dot_product_attention_basic(sample_qkv_tensors):
    """Test basic functionality without masks or dropout."""
    query, key, value = sample_qkv_tensors
    output = scaled_dot_product_attention(query, key, value)

    assert output.shape == query.shape
    # Basic check: output should not be all zeros unless inputs are
    assert not torch.allclose(output, torch.zeros_like(output))


def test_scaled_dot_product_attention_causal_mask(sample_qkv_tensors):
    """Test with causal masking."""
    query, key, value = sample_qkv_tensors
    output = scaled_dot_product_attention(query, key, value, is_causal=True)

    assert output.shape == query.shape

    # More rigorous check: ensure future information is masked
    # For a causal mask, query at position i should only attend to keys at positions <= i.
    # This is hard to test directly without re-implementing SDPA logic.
    # A simpler check: if seq_len_q == seq_len_k, the last query should only attend to itself.
    # This test is more about ensuring the flag is processed.
    if query.shape[-2] == key.shape[-2]:
        # Manually compute attention scores for a single head/batch item
        # and verify the effect of causal mask.
        q_single = query[0, 0, :, :]
        k_single = key[0, 0, :, :]
        v_single = value[0, 0, :, :]

        scores = torch.matmul(q_single, k_single.transpose(-2, -1)) / math.sqrt(query.size(-1))
        causal_mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
        scores_masked = scores.masked_fill(causal_mask, -torch.inf)
        weights = F.softmax(scores_masked, dim=-1)
        expected_output_single = torch.matmul(weights, v_single)

        # Extract corresponding output from the function call
        actual_output_single = output[0, 0, :, :]

        assert torch.allclose(actual_output_single, expected_output_single, atol=1e-6)


def test_scaled_dot_product_attention_attn_mask(sample_qkv_tensors):
    """Test with explicit attention mask."""
    query, key, value = sample_qkv_tensors
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    seq_len_k = key.shape[-2]

    # Create a simple padding mask: mask out the last token for all queries
    attn_mask = torch.zeros(batch_size, 1, 1, seq_len_k, dtype=torch.bool)
    attn_mask[:, :, :, -1] = True  # Mask the last key position

    output = scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)

    assert output.shape == query.shape

    # Verify that masked positions have no influence
    # This is complex, but we can check if the output changes when mask is applied
    output_unmasked = scaled_dot_product_attention(query, key, value, attn_mask=None)
    assert not torch.allclose(output, output_unmasked)


def test_scaled_dot_product_attention_combined_mask(sample_qkv_tensors):
    """Test with both causal and explicit attention mask."""
    query, key, value = sample_qkv_tensors
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    seq_len_k = key.shape[-2]

    # Create a simple padding mask
    padding_mask = torch.zeros(batch_size, 1, 1, seq_len_k, dtype=torch.bool)
    padding_mask[:, :, :, -1] = True  # Mask the last key position

    output = scaled_dot_product_attention(query, key, value, attn_mask=padding_mask, is_causal=True)

    assert output.shape == query.shape

    # Verify combined effect by comparing to manual combination
    q_single = query[0, 0, :, :]
    k_single = key[0, 0, :, :]
    v_single = value[0, 0, :, :]

    scores = torch.matmul(q_single, k_single.transpose(-2, -1)) / math.sqrt(head_dim)

    causal_mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
    combined_mask = causal_mask | padding_mask[0, 0, 0, :seq_len_k].expand_as(
        scores
    )  # Expand padding mask to match scores

    scores_masked = scores.masked_fill(combined_mask, -torch.inf)
    weights = F.softmax(scores_masked, dim=-1)
    expected_output_single = torch.matmul(weights, v_single)

    actual_output_single = output[0, 0, :, :]
    assert torch.allclose(actual_output_single, expected_output_single, atol=1e-6)


def test_scaled_dot_product_attention_dropout(sample_qkv_tensors):
    """Test with dropout enabled (in training mode)."""
    query, key, value = sample_qkv_tensors
    dropout_p = 0.5

    # Ensure dropout is active by enabling grad and not setting a fixed seed
    with torch.set_grad_enabled(True):  # Ensure training mode behavior for F.dropout
        output1 = scaled_dot_product_attention(query, key, value, dropout_p=dropout_p)
        output2 = scaled_dot_product_attention(query, key, value, dropout_p=dropout_p)

    assert output1.shape == query.shape
    assert output2.shape == query.shape

    # Outputs should differ due to dropout (unless dropout_p is 0)
    assert not torch.allclose(output1, output2)


def test_scaled_dot_product_attention_custom_scale(sample_qkv_tensors):
    """Test with a custom scale value."""
    query, key, value = sample_qkv_tensors
    custom_scale = 2.0

    output = scaled_dot_product_attention(query, key, value, scale=custom_scale)

    assert output.shape == query.shape

    # Compare with manual calculation using custom scale
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * custom_scale
    attn_weights = F.softmax(attn_scores, dim=-1)
    expected_output = torch.matmul(attn_weights, value)

    assert torch.allclose(output, expected_output, atol=1e-6)


def test_scaled_dot_product_attention_seq_len_one():
    """Test with sequence length of one."""
    batch_size = 1
    num_heads = 1
    seq_len = 1
    head_dim = 4

    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    output = scaled_dot_product_attention(query, key, value)

    assert output.shape == query.shape
    # For seq_len=1, attention should just be identity (no other tokens to attend to)
    assert torch.allclose(output, value, atol=1e-6)


def test_scaled_dot_product_attention_invalid_attn_mask_dtype(sample_qkv_tensors):
    """Test that ValueError is raised for non-boolean attn_mask dtype."""
    query, key, value = sample_qkv_tensors
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    seq_len_k = key.shape[-2]

    # Create a non-boolean mask
    attn_mask_float = torch.randn(batch_size, 1, 1, seq_len_k)

    with pytest.raises(ValueError, match="attn_mask 必须是布尔类型张量"):  # Updated error message
        scaled_dot_product_attention(query, key, value, attn_mask=attn_mask_float)
