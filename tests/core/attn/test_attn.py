import pytest
import torch

from llm.core.attn import Attention


@pytest.fixture
def attention():
    return Attention(hidden_size=64)


@pytest.fixture
def input_tensor():
    return torch.randn(2, 10, 64)  # [batch_size, seq_len, hidden_size]


def test_attention_initialization(attention):
    """Test if attention module is initialized correctly."""
    assert isinstance(attention, Attention)
    assert attention.hidden_size == 64
    assert attention.scale == 8.0  # sqrt(64)
    assert attention.p == 0.1
    assert not attention.is_causal


def test_attention_forward_shape(attention, input_tensor):
    """Test if forward pass maintains correct shape."""
    output = attention(input_tensor)
    assert output.shape == input_tensor.shape


def test_attention_with_mask(attention, input_tensor):
    """Test attention with attention mask."""
    # Create a simple attention mask
    attn_mask = torch.ones(2, 1, 1, 10, dtype=torch.bool)  # [batch_size, 1, 1, seq_len]
    output = attention(input_tensor, attn_mask)
    assert output.shape == input_tensor.shape


def test_attention_gradients(attention, input_tensor):
    """Test if gradients are computed correctly."""
    input_tensor.requires_grad_(True)
    output = attention(input_tensor)
    loss = output.sum()
    loss.backward()
    assert input_tensor.grad is not None
    assert not torch.isnan(input_tensor.grad).any()


def test_attention_causal(attention, input_tensor):
    """Test causal attention."""
    attention.is_causal = True
    output = attention(input_tensor)
    assert output.shape == input_tensor.shape


def test_attention_different_hidden_sizes():
    """Test attention with different hidden sizes."""
    hidden_sizes = [32, 64, 128]
    for hidden_size in hidden_sizes:
        attention = Attention(hidden_size=hidden_size)
        input_tensor = torch.randn(2, 10, hidden_size)
        output = attention(input_tensor)
        assert output.shape == input_tensor.shape


def test_attention_different_batch_sizes(attention):
    """Test attention with different batch sizes."""
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 10, 64)
        output = attention(input_tensor)
        assert output.shape == input_tensor.shape


def test_attention_different_sequence_lengths(attention):
    """Test attention with different sequence lengths."""
    seq_lengths = [5, 10, 20]
    for seq_len in seq_lengths:
        input_tensor = torch.randn(2, seq_len, 64)
        output = attention(input_tensor)
        assert output.shape == input_tensor.shape
