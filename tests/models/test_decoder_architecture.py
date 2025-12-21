import torch

from llm.core.attn.mha import MultiHeadAttention
from llm.core.mlp import MLP
from llm.models.decoder import DecoderModel


def test_gqa_invariants():
    """
    Verify Grouped Query Attention (GQA) structural invariants.
    """
    hidden_size = 128
    num_heads = 16
    num_kv_heads = 4

    model = DecoderModel(
        vocab_size=100, hidden_size=hidden_size, num_layers=1, num_heads=num_heads, num_kv_heads=num_kv_heads
    )

    mha = model.transformer_blocks[0].self_attn
    assert isinstance(mha, MultiHeadAttention)
    assert mha.num_heads == 16
    assert mha.num_kv_heads == 4

    # QKV projection output dimension: (num_heads + 2 * num_kv_heads) * head_dim
    # (16 + 2 * 4) * 8 = 24 * 8 = 192
    assert mha.qkv_proj.out_features == 192


def test_swiglu_configuration():
    """
    Verify that SwiGLU is correctly configured in the MLP layer.
    """
    hidden_size = 64
    model = DecoderModel(
        vocab_size=100, hidden_size=hidden_size, num_layers=1, num_heads=4, use_glu=True, mlp_activation="silu"
    )

    mlp = model.transformer_blocks[0].mlp
    assert isinstance(mlp, MLP)
    assert mlp.use_glu is True
    assert hasattr(mlp, "gate_proj")
    assert isinstance(mlp.gate_proj, torch.nn.Linear)


def test_parameter_count_scaling():
    """
    Verify that model parameter count scales reasonably with layers and hidden size.
    """

    def get_params(m):
        return sum(p.numel() for p in m.parameters())

    model_small = DecoderModel(vocab_size=100, hidden_size=64, num_layers=1, num_heads=4)
    model_large = DecoderModel(vocab_size=100, hidden_size=128, num_layers=1, num_heads=8)

    # Larger hidden size should significantly increase parameters
    assert get_params(model_large) > get_params(model_small)

    model_2layer = DecoderModel(vocab_size=100, hidden_size=64, num_layers=2, num_heads=4)
    # 2 layers should have more params than 1 layer (roughly double the transformer block part)
    assert get_params(model_2layer) > get_params(model_small)
