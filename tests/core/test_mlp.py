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
    include_norm_residual=True,  # New parameter for the helper
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
        include_norm_residual=include_norm_residual, # Pass to MLP
        device=device,
        dtype=dtype,
    )
    mlp.eval()  # Default to eval mode for tests

    return mlp, input_tensor


# --- Test Case 1: Basic Functionality ---
def test_mlp_basic_functionality():
    """Tests basic MLP functionality with default parameters."""
    torch.manual_seed(42)

    # Create MLP with default configuration (include_norm_residual=True)
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
def test_mlp_normalization_options_with_norm_residual(norm_first):
    """Tests MLP with pre-LN vs post-LN configurations when norm/residual is included."""
    torch.manual_seed(42)

    # Test explicitly with include_norm_residual=True
    mlp, input_tensor = create_test_mlp(norm_first=norm_first, include_norm_residual=True)

    with torch.no_grad():
        output = mlp(input_tensor)

    # Basic shape test
    assert output.shape == input_tensor.shape
    assert mlp.norm is not None, "Norm layer should exist"

    # Create another MLP with opposite norm_first setting to compare
    opposite_mlp, _ = create_test_mlp(norm_first=not norm_first, include_norm_residual=True)
    
    # Ensure weights are identical for a fair comparison of norm_first effect
    # This requires copying state dict carefully if architectures are compatible
    # For MLP, changing norm_first doesn't change layers, just their order of ops with norm
    opposite_mlp.load_state_dict(mlp.state_dict())


    with torch.no_grad():
        opposite_output = opposite_mlp(input_tensor)

    # Outputs should differ between pre-LN and post-LN if hidden_size is not 1 (norm makes a diff)
    if input_tensor.shape[-1] > 1:
         assert not torch.allclose(output, opposite_output, atol=1e-5), \
             f"Output with norm_first={norm_first} should differ from norm_first={not norm_first} when norm/residual is active"
    else:
        # If hidden_size is 1, LayerNorm might not change values much or at all.
        # This part of the test is less meaningful for hidden_size=1.
        pass


# --- Test Case for include_norm_residual=False ---
def test_mlp_no_norm_residual_output():
    """Tests MLP behavior when include_norm_residual is False."""
    torch.manual_seed(42)
    hidden_size_test = 64
    mlp_no_norm_res, input_tensor = create_test_mlp(
        hidden_size=hidden_size_test,
        include_norm_residual=False,
        dropout_p=0.0 # Disable dropout for exact output check
    )
    mlp_no_norm_res.eval()

    assert mlp_no_norm_res.norm is None, "Norm layer should not exist when include_norm_residual=False"
    assert mlp_no_norm_res.include_norm_residual is False

    with torch.no_grad():
        output_no_norm_res = mlp_no_norm_res(input_tensor)

    # Manually compute expected output: fc1 -> activation -> fc2
    manual_output = mlp_no_norm_res.fc1(input_tensor)
    manual_output = mlp_no_norm_res.activation(manual_output)
    # Dropout is 0.0 and in eval mode, so self.dropout(manual_output) is manual_output
    manual_output = mlp_no_norm_res.fc2(manual_output)

    assert torch.allclose(output_no_norm_res, manual_output, atol=1e-6), \
        "Output with no norm/residual does not match manual computation."

    # Compare with MLP that *does* include norm and residual
    mlp_with_norm_res, _ = create_test_mlp(
        hidden_size=hidden_size_test,
        include_norm_residual=True,
        dropout_p=0.0
    )
    mlp_with_norm_res.eval()
    # Copy weights for fair comparison
    # Note: mlp_with_norm_res has a 'norm' layer, mlp_no_norm_res does not.
    # So strict=False is needed for load_state_dict.
    # We are primarily interested in fc1, fc2, activation weights.
    # A better way is to copy layer by layer if using parts of the state_dict.
    # For simplicity, we'll just ensure they compute differently.
    # A full state dict copy won't work directly if `norm` parameters are present in one but not other.
    # Let's re-initialize mlp_with_norm_res and copy relevant weights
    mlp_with_norm_res.fc1.load_state_dict(mlp_no_norm_res.fc1.state_dict())
    mlp_with_norm_res.fc2.load_state_dict(mlp_no_norm_res.fc2.state_dict())
    # Activation layers typically don't have state, but if they did:
    # mlp_with_norm_res.activation.load_state_dict(mlp_no_norm_res.activation.state_dict())


    with torch.no_grad():
        output_with_norm_res = mlp_with_norm_res(input_tensor)
    
    # Outputs should generally differ if norm/residual make a difference
    if hidden_size_test > 1: # LayerNorm typically changes things if hidden_size > 1
        assert not torch.allclose(output_no_norm_res, output_with_norm_res, atol=1e-5), \
            "Output with no norm/residual should differ from output with norm/residual."

def test_mlp_no_norm_residual_dropout_still_active():
    """Tests that dropout is still active in MLP when include_norm_residual is False."""
    torch.manual_seed(42)
    dropout_val = 0.5
    mlp, input_tensor = create_test_mlp(
        include_norm_residual=False,
        dropout_p=dropout_val
    )

    # Test in eval mode (dropout should be disabled)
    mlp.eval()
    with torch.no_grad():
        output_eval_1 = mlp(input_tensor)
        output_eval_2 = mlp(input_tensor)
    assert torch.allclose(output_eval_1, output_eval_2, atol=1e-7), \
        "Outputs in eval mode should be identical (no norm/res)."

    # Test in train mode (dropout should be active)
    mlp.train()
    # Ensure different random seeds for dropout mask generation if not handled by PyTorch's global seed already
    # However, multiple calls to dropout in train() should yield different masks for the same input.
    torch.manual_seed(43) # Change seed slightly for first train output if necessary
    with torch.no_grad():
        output_train_1 = mlp(input_tensor)
    torch.manual_seed(44) # And for second
    with torch.no_grad():
        output_train_2 = mlp(input_tensor)
    
    if dropout_val > 0:
        assert not torch.allclose(output_train_1, output_train_2, atol=1e-6), \
            "Outputs in train mode should differ due to dropout (no norm/res)."


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

    # Test in train mode (dropout should be active) - this part is for default MLP with norm/res
    mlp_default_dropout, input_tensor_default = create_test_mlp(dropout_p=0.5, include_norm_residual=True)
    mlp_default_dropout.train()
    torch.manual_seed(45)
    with torch.no_grad():
        output_train_1_default = mlp_default_dropout(input_tensor_default)
    torch.manual_seed(46)
    with torch.no_grad():
        output_train_2_default = mlp_default_dropout(input_tensor_default)

    if 0.5 > 0:
        assert not torch.allclose(output_train_1_default, output_train_2_default, atol=1e-6), \
            "Outputs in train mode should differ due to dropout (default MLP with norm/res)"


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
