import math

import pytest
import torch

from llm.core.attn import MultiLatentAttention


@pytest.fixture
def mla():
    return MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=16)


@pytest.fixture
def input_tensor():
    return torch.randn(2, 10, 64)  # [batch_size, seq_len, hidden_size]


def test_mla_initialization(mla):
    """Test if MLA module is initialized correctly."""
    assert isinstance(mla, MultiLatentAttention)
    assert mla.hidden_size == 64
    assert mla.num_heads == 8
    assert mla.head_dim == 8  # 64 // 8
    torch.testing.assert_close(mla.scale, 1 / math.sqrt(8), atol=1e-12, rtol=1e-12)
    assert mla.dropout_p == 0.1
    assert not mla.is_causal
    assert mla.num_latents == 16
    assert mla.latent_dim == 64


def test_mla_forward_shape(mla, input_tensor):
    """Test if forward pass maintains correct shape."""
    output = mla(input_tensor)
    assert output.shape == input_tensor.shape


def test_mla_with_mask(mla, input_tensor):
    """Test MLA with attention mask."""
    # Create a simple attention mask
    attn_mask = torch.ones(2, 1, 1, 10, dtype=torch.bool)  # [batch_size, 1, 1, seq_len]
    output = mla(input_tensor, attn_mask)
    assert output.shape == input_tensor.shape


def test_mla_gradients(mla, input_tensor):
    """Test if gradients are computed correctly."""
    input_tensor.requires_grad_(True)
    output = mla(input_tensor)
    loss = output.sum()
    loss.backward()
    assert input_tensor.grad is not None
    assert not torch.isnan(input_tensor.grad).any()


def test_mla_causal(mla, input_tensor):
    """Test causal MLA."""
    mla.is_causal = True
    output = mla(input_tensor)
    assert output.shape == input_tensor.shape


def test_mla_different_num_latents():
    """Test MLA with different number of latents."""
    num_latents_list = [8, 16, 32]
    for num_latents in num_latents_list:
        mla = MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=num_latents)
        input_tensor = torch.randn(2, 10, 64)
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


def test_mla_different_latent_dims():
    """Test MLA with different latent dimensions."""
    latent_dims = [32, 64, 128]
    for latent_dim in latent_dims:
        mla = MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=16, latent_dim=latent_dim)
        # Use the same hidden_size for input tensor
        input_tensor = torch.randn(2, 10, 64)  # hidden_size should match the model's hidden_size
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


def test_mla_different_hidden_sizes():
    """Test MLA with different hidden sizes."""
    hidden_sizes = [32, 64, 128]
    for hidden_size in hidden_sizes:
        mla = MultiLatentAttention(hidden_size=hidden_size, num_heads=8, num_latents=16)
        input_tensor = torch.randn(2, 10, hidden_size)
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


def test_mla_different_batch_sizes(mla):
    """Test MLA with different batch sizes."""
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 10, 64)
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


def test_mla_different_sequence_lengths(mla):
    """Test MLA with different sequence lengths."""
    seq_lengths = [5, 10, 20]
    for seq_len in seq_lengths:
        input_tensor = torch.randn(2, seq_len, 64)
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


def test_mla_norm_first():
    """Test MLA with norm_first=True."""
    mla = MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=16, norm_first=True)
    input_tensor = torch.randn(2, 10, 64)
    output = mla(input_tensor)
    assert output.shape == input_tensor.shape
