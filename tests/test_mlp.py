import pytest
import torch

from llm.core.mlp import MLP


# Helper function to create MLP and input tensor
def _create_mlp_input(
    batch_size, seq_len, hidden_size, intermediate_factor, activation, use_layer_norm, bias, dropout_p, device, dtype
):
    intermediate_size = hidden_size * intermediate_factor
    if seq_len is None:
        input_shape = (batch_size, hidden_size)
    else:
        input_shape = (batch_size, seq_len, hidden_size)
    input_tensor = torch.randn(input_shape, dtype=dtype, device=device)

    mlp = MLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        dropout_p=dropout_p,
        use_layer_norm=use_layer_norm,
        bias=bias,
        device=device,
        dtype=dtype,
    )
    mlp.eval()  # Default to eval mode for tests
    return mlp, input_tensor, intermediate_size


# --- Test Case 1: Output Shape ---
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [None, 10])  # None for 2D input, value for 3D
@pytest.mark.parametrize("hidden_size", [64])
def test_mlp_output_shape(batch_size, seq_len, hidden_size):
    """Tests if the MLP output shape matches the input shape."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    mlp, input_tensor, _ = _create_mlp_input(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_factor=4,
        activation="gelu",
        use_layer_norm=True,
        bias=True,
        dropout_p=0.0,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        output = mlp(input_tensor)

    assert output.shape == input_tensor.shape, (
        f"Output shape {output.shape} does not match input shape {input_tensor.shape}"
    )


# --- Test Case 2: Transformation Effect (Output != Input) ---
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [None, 10])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("intermediate_factor", [2])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("use_layer_norm", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_mlp_transformation_effect(
    batch_size, seq_len, hidden_size, intermediate_factor, activation, use_layer_norm, bias
):
    """Tests if the MLP output is generally different from the input."""
    torch.manual_seed(43)  # Use different seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    mlp, input_tensor, intermediate_size = _create_mlp_input(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_factor=intermediate_factor,
        activation=activation,
        use_layer_norm=use_layer_norm,
        bias=bias,
        dropout_p=0.0,
        device=device,
        dtype=dtype,  # Keep dropout 0 for simplicity
    )
    input_tensor_clone = input_tensor.clone()

    with torch.no_grad():
        output = mlp(input_tensor)

    if intermediate_size > 0:
        assert not torch.allclose(output, input_tensor_clone, atol=1e-6), (
            f"Output tensor is unexpectedly identical to the input tensor for config: "
            f"use_layer_norm={use_layer_norm}, activation={activation}, bias={bias}"
        )


# --- Test Case 3: Pre-LayerNorm Effect ---
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [None, 10])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("intermediate_factor", [2])
@pytest.mark.parametrize("activation", ["gelu"])
@pytest.mark.parametrize("bias", [True, False])
def test_mlp_pre_ln_effect(batch_size, seq_len, hidden_size, intermediate_factor, activation, bias):
    """Tests that enabling Pre-LayerNorm changes the output."""
    torch.manual_seed(44)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    dropout_p = 0.0  # Keep dropout 0

    # MLP with LayerNorm
    mlp_ln, input_tensor, intermediate_size = _create_mlp_input(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_factor=intermediate_factor,
        activation=activation,
        use_layer_norm=True,
        bias=bias,
        dropout_p=dropout_p,
        device=device,
        dtype=dtype,
    )
    input_tensor_clone = input_tensor.clone()

    # MLP without LayerNorm (same weights)
    mlp_no_ln, _, _ = _create_mlp_input(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_factor=intermediate_factor,
        activation=activation,
        use_layer_norm=False,
        bias=bias,
        dropout_p=dropout_p,
        device=device,
        dtype=dtype,
    )
    mlp_no_ln.load_state_dict(mlp_ln.state_dict(), strict=False)

    with torch.no_grad():
        output_ln = mlp_ln(input_tensor)
        output_no_ln = mlp_no_ln(input_tensor_clone)

    if intermediate_size > 0:
        assert not torch.allclose(output_ln, output_no_ln, atol=1e-6), (
            f"Output with LayerNorm is unexpectedly identical to output without LayerNorm for config: "
            f"activation={activation}, bias={bias}"
        )


# --- Test Case 4: Bias Effect ---
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [None, 10])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("intermediate_factor", [2])
@pytest.mark.parametrize("activation", ["gelu"])
@pytest.mark.parametrize("use_layer_norm", [True, False])
def test_mlp_bias_parameter_existence(
    batch_size, seq_len, hidden_size, intermediate_factor, activation, use_layer_norm
):
    """Tests that the bias parameter exists or not based on the bias flag."""
    torch.manual_seed(45)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    dropout_p = 0.0  # Keep dropout 0

    # MLP with Bias
    mlp_bias, _, _ = _create_mlp_input(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_factor=intermediate_factor,
        activation=activation,
        use_layer_norm=use_layer_norm,
        bias=True,
        dropout_p=dropout_p,
        device=device,
        dtype=dtype,
    )

    # MLP without Bias
    mlp_no_bias, _, _ = _create_mlp_input(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_factor=intermediate_factor,
        activation=activation,
        use_layer_norm=use_layer_norm,
        bias=False,
        dropout_p=dropout_p,
        device=device,
        dtype=dtype,
    )

    # Check bias parameter existence
    assert mlp_bias.fc1.bias is not None, "fc1.bias should exist when bias=True"
    assert mlp_bias.fc2.bias is not None, "fc2.bias should exist when bias=True"
    assert mlp_no_bias.fc1.bias is None, "fc1.bias should be None when bias=False"
    assert mlp_no_bias.fc2.bias is None, "fc2.bias should be None when bias=False"


# --- Test Case 5: Dropout Effect in Eval Mode ---
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [None, 10])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("intermediate_factor", [2])
@pytest.mark.parametrize("activation", ["gelu"])
@pytest.mark.parametrize("use_layer_norm", [True, False])
@pytest.mark.parametrize("bias", [True])
def test_mlp_dropout_in_eval_mode(
    batch_size, seq_len, hidden_size, intermediate_factor, activation, use_layer_norm, bias
):
    """Tests that dropout has no effect in eval mode."""
    torch.manual_seed(46)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # MLP with Dropout (p=0.5)
    mlp_dropout, input_tensor, intermediate_size = _create_mlp_input(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_factor=intermediate_factor,
        activation=activation,
        use_layer_norm=use_layer_norm,
        bias=bias,
        dropout_p=0.5,
        device=device,
        dtype=dtype,  # High dropout prob
    )
    input_tensor_clone = input_tensor.clone()

    # MLP without Dropout (p=0.0) - should have same weights/buffers after loading state dict
    mlp_no_dropout, _, _ = _create_mlp_input(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_factor=intermediate_factor,
        activation=activation,
        use_layer_norm=use_layer_norm,
        bias=bias,
        dropout_p=0.0,
        device=device,
        dtype=dtype,
    )
    # Explicitly load state dict to ensure weights/buffers are identical.
    # Use strict=True because the underlying architecture (layers) is the same.
    mlp_no_dropout.load_state_dict(mlp_dropout.state_dict(), strict=True)
    # Ensure both are in eval mode
    mlp_dropout.eval()
    mlp_no_dropout.eval()

    with torch.no_grad():
        output_dropout = mlp_dropout(input_tensor)
        output_no_dropout = mlp_no_dropout(input_tensor_clone)

    if intermediate_size > 0:
        # In eval mode, dropout is off, so outputs should be identical
        assert torch.allclose(output_dropout, output_no_dropout, atol=1e-7), (
            f"Output with dropout={mlp_dropout.dropout.p} is unexpectedly different from output with dropout=0.0 in eval mode for config: "
            f"activation={activation}, use_layer_norm={use_layer_norm}, bias={bias}"
        )
