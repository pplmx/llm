import pytest
import torch

from llm.core.mlp import MLP


# Helper function to create MLP and input tensor with more focused parameters
def create_test_mlp(
    hidden_size=64,
    intermediate_factor=4,
    activation="gelu",
    norm_first=True,
    bias=True,
    dropout_p=0.0,
    input_dims=None,
    device=None,
    dtype=None,
):
    """
    Creates an MLP instance and input tensor for testing.

    Args:
        hidden_size: Dimensionality of input/output features
        intermediate_factor: Multiplier for intermediate size
        activation: Activation function
        norm_first: Whether to use pre-LN (True) or post-LN (False)
        bias: Whether to use bias in linear layers
        dropout_p: Dropout probability
        input_dims: Additional input dimensions (e.g., [batch_size, seq_len])
        device: torch device
        dtype: torch dtype

    Returns:
        mlp: Configured MLP instance in eval mode
        input_tensor: Input tensor with appropriate shape
    """
    # Set defaults for device and dtype
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32

    # Calculate actual intermediate size
    intermediate_size = hidden_size * intermediate_factor

    # Create input tensor with appropriate shape
    if input_dims is None:
        input_dims = [2, 8]  # Default batch_size=2, seq_len=8

    # Input shape includes hidden_size as the last dimension
    input_shape = input_dims + [hidden_size]
    input_tensor = torch.randn(input_shape, dtype=dtype, device=device)

    # Create and configure MLP
    mlp = MLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        dropout_p=dropout_p,
        norm_first=norm_first,
        bias=bias,
        device=device,
        dtype=dtype,
    )
    mlp.eval()  # Default to eval mode for tests

    return mlp, input_tensor


# --- Test Case 1: Basic Functionality ---
def test_mlp_basic_functionality():
    """Tests basic MLP functionality with default parameters."""
    torch.manual_seed(42)

    # Create MLP with default configuration
    mlp, input_tensor = create_test_mlp()

    # Run forward pass
    with torch.no_grad():
        output = mlp(input_tensor)

    # Check output shape matches input shape
    assert output.shape == input_tensor.shape, (
        f"Output shape {output.shape} does not match input shape {input_tensor.shape}"
    )

    # Verify output differs from input (transformation happened)
    assert not torch.allclose(output, input_tensor, atol=1e-6), "Output tensor should not be identical to input tensor"


# --- Test Case 2: Different Input Shapes ---
@pytest.mark.parametrize(
    "input_dims",
    [
        [1],  # Just batch size (2D input)
        [2, 10],  # Batch size and sequence length (3D input)
        [3, 4, 5],  # Batch size, dim1, dim2 (4D input)
    ],
)
def test_mlp_different_input_shapes(input_dims):
    """Tests MLP with various input shapes."""
    torch.manual_seed(42)

    mlp, input_tensor = create_test_mlp(input_dims=input_dims)

    with torch.no_grad():
        output = mlp(input_tensor)

    assert output.shape == input_tensor.shape, (
        f"Output shape {output.shape} does not match input shape {input_tensor.shape}"
    )


# --- Test Case 3: Normalization Options ---
@pytest.mark.parametrize("norm_first", [True, False])
def test_mlp_normalization_options(norm_first):
    """Tests MLP with pre-LN vs post-LN configurations."""
    torch.manual_seed(42)

    mlp, input_tensor = create_test_mlp(norm_first=norm_first)

    with torch.no_grad():
        output = mlp(input_tensor)

    # Basic shape test
    assert output.shape == input_tensor.shape

    # Create another MLP with opposite norm_first setting to compare
    opposite_mlp, _ = create_test_mlp(norm_first=not norm_first)

    # Copy weights where possible
    opposite_mlp.load_state_dict(mlp.state_dict(), strict=False)

    with torch.no_grad():
        opposite_output = opposite_mlp(input_tensor)

    # Outputs should differ between pre-LN and post-LN
    assert not torch.allclose(output, opposite_output, atol=1e-6), (
        f"Output with norm_first={norm_first} should differ from norm_first={not norm_first}"
    )


# --- Test Case 4: Activation Functions ---
@pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
def test_mlp_activation_functions(activation):
    """Tests MLP with different activation functions."""
    torch.manual_seed(42)

    mlp, input_tensor = create_test_mlp(activation=activation)

    with torch.no_grad():
        output = mlp(input_tensor)

    # Basic shape test
    assert output.shape == input_tensor.shape


# --- Test Case 5: Bias Parameter ---
def test_mlp_bias_parameter():
    """Tests bias parameter existence based on bias flag."""
    torch.manual_seed(42)

    # MLP with bias
    mlp_with_bias, _ = create_test_mlp(bias=True)

    # MLP without bias
    mlp_without_bias, _ = create_test_mlp(bias=False)

    # Check bias parameter existence
    assert mlp_with_bias.fc1.bias is not None, "fc1.bias should exist when bias=True"
    assert mlp_with_bias.fc2.bias is not None, "fc2.bias should exist when bias=True"
    assert mlp_without_bias.fc1.bias is None, "fc1.bias should be None when bias=False"
    assert mlp_without_bias.fc2.bias is None, "fc2.bias should be None when bias=False"


# --- Test Case 6: Dropout Behavior ---
def test_mlp_dropout_behavior():
    """Tests dropout behavior in train vs eval modes."""
    torch.manual_seed(42)

    # Create MLP with high dropout
    mlp, input_tensor = create_test_mlp(dropout_p=0.5)

    # Test in eval mode (dropout should be disabled)
    mlp.eval()
    with torch.no_grad():
        output_eval_1 = mlp(input_tensor)
        output_eval_2 = mlp(input_tensor)

    # Outputs in eval mode should be identical (deterministic)
    assert torch.allclose(output_eval_1, output_eval_2, atol=1e-7), (
        "Outputs in eval mode should be identical with fixed input"
    )

    # Test in train mode (dropout should be active)
    mlp.train()
    with torch.no_grad():  # We still don't need gradients for this test
        output_train_1 = mlp(input_tensor)
        output_train_2 = mlp(input_tensor)

    # Outputs in train mode should differ due to random dropout
    assert not torch.allclose(output_train_1, output_train_2, atol=1e-6), (
        "Outputs in train mode should differ due to dropout"
    )


# --- Test Case 7: Hidden Size and Intermediate Size ---
@pytest.mark.parametrize(
    "hidden_size,intermediate_factor",
    [
        (32, 2),  # Small hidden, small expansion
        (64, 4),  # Medium hidden, standard expansion
        (128, 8),  # Large hidden, large expansion
    ],
)
def test_mlp_size_configurations(hidden_size, intermediate_factor):
    """Tests different hidden size and intermediate size configurations."""
    torch.manual_seed(42)

    mlp, input_tensor = create_test_mlp(hidden_size=hidden_size, intermediate_factor=intermediate_factor)

    # Verify the intermediate size is correctly configured
    assert mlp.intermediate_size == hidden_size * intermediate_factor

    with torch.no_grad():
        output = mlp(input_tensor)

    # Check shape consistency
    assert output.shape == input_tensor.shape
