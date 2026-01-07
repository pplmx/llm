import pytest
import torch

from llm.core.attn.sdpa import sdpa


@pytest.fixture
def sample_qkv_tensors():
    """Create sample Q, K, V tensors for testing."""
    batch_size = 2
    num_heads = 4
    seq_len = 8
    head_dim = 16
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    return q, k, v


def test_sdpa_basic(sample_qkv_tensors):
    """Test basic functionality without masks."""
    q, k, v = sample_qkv_tensors
    output = sdpa(q, k, v)
    assert output.shape == q.shape


def test_sdpa_with_attn_mask_bool(sample_qkv_tensors):
    """Test with boolean attention mask (True=Mask Out)."""
    q, k, v = sample_qkv_tensors
    batch_size, num_heads, seq_len, _ = q.shape

    # Mask last token
    mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.bool)
    mask[..., -1] = True

    output = sdpa(q, k, v, attn_mask=mask)
    assert output.shape == q.shape


def test_sdpa_with_attn_mask_float(sample_qkv_tensors):
    """Test with float attention mask (0/-inf)."""
    q, k, v = sample_qkv_tensors
    batch_size, num_heads, seq_len, _ = q.shape

    # Mask last token with large negative value
    mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.float32)
    mask[..., -1] = float("-inf")

    output = sdpa(q, k, v, attn_mask=mask)
    assert output.shape == q.shape


def test_sdpa_window_size(sample_qkv_tensors):
    """Test window_size functionality."""
    q, k, v = sample_qkv_tensors
    output = sdpa(q, k, v, window_size=2)
    assert output.shape == q.shape


def test_sdpa_causal(sample_qkv_tensors):
    """Test causal masking."""
    q, k, v = sample_qkv_tensors
    output = sdpa(q, k, v, is_causal=True)
    assert output.shape == q.shape


def test_sdpa_causal_and_window(sample_qkv_tensors):
    """Test combined causal and window masking."""
    q, k, v = sample_qkv_tensors
    output = sdpa(q, k, v, is_causal=True, window_size=2)
    assert output.shape == q.shape
