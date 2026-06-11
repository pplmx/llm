import pytest
import torch

from llm.core.attn.mha import MultiHeadAttention
from llm.core.mlp import MLP
from llm.core.rms_norm import RMSNorm
from llm.models.decoder import DecoderModel


@pytest.mark.slow
def test_swiglu_mlp():
    hidden_size = 64
    intermediate_size = 128
    # SwiGLU typically uses SiLU (silu)
    mlp = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size, activation="silu", use_glu=True)
    x = torch.randn(2, 10, hidden_size)
    output = mlp(x)
    assert output.shape == x.shape
    # Check if gate_proj exists
    assert hasattr(mlp, "gate_proj")


@pytest.mark.slow
def test_gqa_attention():
    hidden_size = 64
    num_heads = 8
    num_kv_heads = 2  # GQA
    mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads)
    x = torch.randn(2, 10, hidden_size)
    output = mha(x)
    assert output.shape == x.shape
    # Check if k_proj and q_proj have different output dims
    # Verify projections exist and have correct sizes
    assert hasattr(mha, "qkv_proj")
    # qkv_dim = (num_heads + 2*num_kv_heads) * head_dim = (8 + 2*2) * (64//8) = 12 * 8 = 96
    assert mha.qkv_proj.out_features == 96


@pytest.mark.slow
def test_rmsnorm_decoder():
    # Use RMSNorm in DecoderModel
    vocab_size = 100
    hidden_size = 64
    model = DecoderModel(
        vocab_size=vocab_size, hidden_size=hidden_size, num_layers=1, num_heads=4, norm_impl="rms_norm"
    )
    x = torch.randint(0, vocab_size, (2, 10))
    output = model(x)
    assert output.shape == (2, 10, vocab_size)
    # Check if final_norm is RMSNorm
    assert isinstance(model.final_norm, RMSNorm)
    # Check if transformer blocks use RMSNorm
    assert isinstance(model.transformer_blocks[0].norm1, RMSNorm)


@pytest.mark.slow
def test_gqa_kv_cache():
    from llm.core.kv_cache import KVCache

    hidden_size = 64
    num_heads = 4
    num_kv_heads = 1
    head_dim = hidden_size // num_heads
    mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads)

    kv_cache = KVCache(max_batch_size=1, max_seq_len=16, num_kv_heads=num_kv_heads, head_dim=head_dim)
    x_initial = torch.randn(1, 5, hidden_size)
    _output_initial, _ = mha(x_initial, kv_cache=kv_cache, use_cache=True)

    x_next = torch.randn(1, 1, hidden_size)
    output_next, _ = mha(x_next, kv_cache=kv_cache, use_cache=True)

    assert output_next.shape == (1, 1, hidden_size)
    assert kv_cache.seq_len == 6
