import pytest
import torch
import torch.nn as nn

from llm.core.mlp import MLP


@pytest.mark.slow
def test_mlp_swiglu_initialization():
    """Test SwiGLU initialization logic."""
    hidden_size = 64
    intermediate_size = 128

    # Test with GLU
    mlp_glu = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size, use_glu=True)
    assert hasattr(mlp_glu, "gate_proj")
    assert isinstance(mlp_glu.gate_proj, nn.Linear)
    assert mlp_glu.gate_proj.out_features == intermediate_size

    # Test without GLU
    mlp_no_glu = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size, use_glu=False)
    assert not hasattr(mlp_no_glu, "gate_proj")


@pytest.mark.slow
def test_mlp_swiglu_forward():
    """Test SwiGLU forward pass with silu."""
    hidden_size = 64
    intermediate_size = 128
    mlp = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size, activation="silu", use_glu=True)
    mlp.eval()

    x = torch.randn(2, 5, hidden_size)
    with torch.no_grad():
        output = mlp(x)

    assert output.shape == (2, 5, hidden_size)


@pytest.mark.slow
def test_mlp_swiglu_with_varied_activations():
    """Test SwiGLU with different activations."""
    hidden_size = 32
    intermediate_size = 64

    for act in ["silu", "gelu", "relu"]:
        mlp = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size, activation=act, use_glu=True)
        x = torch.randn(1, 4, hidden_size)
        output = mlp(x)
        assert output.shape == (1, 4, hidden_size)


@pytest.mark.slow
def test_mlp_numerical_consistency_glu():
    """Verify GLU manual computation consistency."""
    hidden_size = 16
    intermediate_size = 32
    mlp = MLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="relu",
        use_glu=True,
        dropout_p=0.0,
        include_norm_residual=False,
    )
    mlp.eval()

    x = torch.randn(1, 1, hidden_size)
    with torch.no_grad():
        # Manual computation: (act(fc1(x)) * gate_proj(x)) -> fc2
        res_fc1 = mlp.activation(mlp.fc1(x))
        res_gate = mlp.gate_proj(x)
        expected = mlp.fc2(res_fc1 * res_gate)

        actual = mlp(x)
        assert torch.allclose(actual, expected, atol=1e-6)
