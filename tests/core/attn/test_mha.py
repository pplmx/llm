import pytest
import torch

from llm.core.attn import MultiHeadAttention


@pytest.fixture
def mha():
    return MultiHeadAttention(hidden_size=64, num_heads=8)


@pytest.fixture
def input_tensor():
    return torch.randn(2, 10, 64)  # [batch_size, seq_len, hidden_size]


def test_mha_initialization(mha):
    """Test if MHA module is initialized correctly."""
    assert isinstance(mha, MultiHeadAttention)
    assert mha.hidden_size == 64
    assert mha.num_heads == 8
    assert mha.head_dim == 8  # 64 // 8
    assert mha.scale == 2.8284271247461903  # sqrt(8)
    assert mha.p == 0.1
    assert not mha.is_causal


def test_mha_forward_shape(mha, input_tensor):
    """Test if forward pass maintains correct shape."""
    output = mha(input_tensor)
    assert output.shape == input_tensor.shape


def test_mha_with_mask(mha, input_tensor):
    """Test MHA with attention mask."""
    # Create a simple attention mask
    attn_mask = torch.ones(2, 1, 1, 10, dtype=torch.bool)  # [batch_size, 1, 1, seq_len]
    output = mha(input_tensor, attn_mask)
    assert output.shape == input_tensor.shape


def test_mha_gradients(mha, input_tensor):
    """Test if gradients are computed correctly."""
    input_tensor.requires_grad_(True)
    output = mha(input_tensor)
    loss = output.sum()
    loss.backward()
    assert input_tensor.grad is not None
    assert not torch.isnan(input_tensor.grad).any()


def test_mha_causal(mha, input_tensor):
    """Test causal MHA."""
    mha.is_causal = True
    output = mha(input_tensor)
    assert output.shape == input_tensor.shape


def test_mha_different_num_heads():
    """Test MHA with different number of heads."""
    num_heads_list = [2, 4, 8]
    for num_heads in num_heads_list:
        mha = MultiHeadAttention(hidden_size=64, num_heads=num_heads)
        input_tensor = torch.randn(2, 10, 64)
        output = mha(input_tensor)
        assert output.shape == input_tensor.shape


def test_mha_different_hidden_sizes():
    """Test MHA with different hidden sizes."""
    hidden_sizes = [32, 64, 128]
    for hidden_size in hidden_sizes:
        mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=8)
        input_tensor = torch.randn(2, 10, hidden_size)
        output = mha(input_tensor)
        assert output.shape == input_tensor.shape


def test_mha_different_batch_sizes(mha):
    """Test MHA with different batch sizes."""
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 10, 64)
        output = mha(input_tensor)
        assert output.shape == input_tensor.shape


def test_mha_different_sequence_lengths(mha):
    """Test MHA with different sequence lengths."""
    seq_lengths = [5, 10, 20]
    for seq_len in seq_lengths:
        input_tensor = torch.randn(2, seq_len, 64)
        output = mha(input_tensor)
        assert output.shape == input_tensor.shape


def test_mha_norm_first():
    """Test MHA with norm_first=True."""
    mha = MultiHeadAttention(hidden_size=64, num_heads=8, norm_first=True)
    input_tensor = torch.randn(2, 10, 64)
    output = mha(input_tensor)
    assert output.shape == input_tensor.shape
