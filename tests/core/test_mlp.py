import pytest
import torch
from torch import nn  # Added nn

from llm.core.mlp import MLP


# Helper function to create MLP and input tensor with more focused parameters
def create_test_mlp(
    hidden_size=64,
    intermediate_factor=4,
    activation="gelu",
    norm_first=True,
    norm_type: type[nn.Module] | nn.Module = nn.LayerNorm,  # Added norm_type
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
        norm_type=norm_type,  # Pass norm_type
        bias=bias,
        include_norm_residual=include_norm_residual,  # Pass to MLP
        device=device,
        dtype=dtype,
    )
    mlp.eval()  # Default to eval mode for tests

    return mlp, input_tensor


# --- Test Case 1: Basic Functionality ---
@pytest.mark.parametrize("bias_val", [True, False])
@pytest.mark.slow
def test_mlp_basic_functionality(bias_val):
    """Tests basic MLP functionality with default and no-bias parameters."""
    torch.manual_seed(42)

    mlp, input_tensor = create_test_mlp(bias=bias_val)

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
@pytest.mark.slow
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
@pytest.mark.parametrize("norm_first_val", [True, False])
@pytest.mark.parametrize("bias_val", [True, False])
@pytest.mark.slow
def test_mlp_normalization_options_with_norm_residual(norm_first_val, bias_val):
    """
    Tests MLP with pre-LN vs post-LN configurations when norm/residual is included,
    also considering bias.
    """
    torch.manual_seed(42)

    # Test explicitly with include_norm_residual=True
    mlp, input_tensor = create_test_mlp(norm_first=norm_first_val, include_norm_residual=True, bias=bias_val)

    with torch.no_grad():
        output = mlp(input_tensor)

    # Basic shape test
    assert output.shape == input_tensor.shape
    assert mlp.norm is not None, "Norm layer should exist"

    # Create another MLP with opposite norm_first setting to compare
    opposite_mlp, _ = create_test_mlp(
        norm_first=not norm_first_val,
        include_norm_residual=True,
        bias=bias_val,  # Keep bias consistent for this comparison
    )

    # Ensure weights are identical for a fair comparison of norm_first effect
    # This requires copying state dict carefully if architectures are compatible
    # For MLP, changing norm_first doesn't change layers, just their order of ops with norm
    opposite_mlp.load_state_dict(mlp.state_dict())

    with torch.no_grad():
        opposite_output = opposite_mlp(input_tensor)

    # Outputs should differ between pre-LN and post-LN if hidden_size is not 1 (norm makes a diff)
    if input_tensor.shape[-1] > 1:
        assert not torch.allclose(output, opposite_output, atol=1e-5), (
            f"Output with norm_first={norm_first_val} should differ from norm_first={not norm_first_val} "
            f"when norm/residual is active (bias={bias_val})"
        )
    else:
        # If hidden_size is 1, LayerNorm might not change values much or at all.
        # This part of the test is less meaningful for hidden_size=1.
        pass


# --- Test Case for include_norm_residual=False ---
@pytest.mark.slow
def test_mlp_no_norm_residual_output():
    """Tests MLP behavior when include_norm_residual is False."""
    torch.manual_seed(42)
    hidden_size_test = 64
    mlp_no_norm_res, input_tensor = create_test_mlp(
        hidden_size=hidden_size_test,
        include_norm_residual=False,
        dropout_p=0.0,  # Disable dropout for exact output check
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

    assert torch.allclose(output_no_norm_res, manual_output, atol=1e-6), (
        "Output with no norm/residual does not match manual computation."
    )

    # Compare with MLP that *does* include norm and residual
    mlp_with_norm_res, _ = create_test_mlp(hidden_size=hidden_size_test, include_norm_residual=True, dropout_p=0.0)
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
    if hidden_size_test > 1:  # LayerNorm typically changes things if hidden_size > 1
        assert not torch.allclose(output_no_norm_res, output_with_norm_res, atol=1e-5), (
            "Output with no norm/residual should differ from output with norm/residual."
        )


@pytest.mark.slow
def test_mlp_no_norm_residual_dropout_still_active():
    """Tests that dropout is still active in MLP when include_norm_residual is False."""
    torch.manual_seed(42)
    dropout_val = 0.5
    mlp, input_tensor = create_test_mlp(include_norm_residual=False, dropout_p=dropout_val)

    # Test in eval mode (dropout should be disabled)
    mlp.eval()
    with torch.no_grad():
        output_eval_1 = mlp(input_tensor)
        output_eval_2 = mlp(input_tensor)
    assert torch.allclose(output_eval_1, output_eval_2, atol=1e-7), (
        "Outputs in eval mode should be identical (no norm/res)."
    )

    # Test in train mode (dropout should be active)
    mlp.train()
    # Ensure different random seeds for dropout mask generation if not handled by PyTorch's global seed already
    # However, multiple calls to dropout in train() should yield different masks for the same input.
    torch.manual_seed(43)  # Change seed slightly for first train output if necessary
    with torch.no_grad():
        output_train_1 = mlp(input_tensor)
    torch.manual_seed(44)  # And for second
    with torch.no_grad():
        output_train_2 = mlp(input_tensor)

    if dropout_val > 0:
        assert not torch.allclose(output_train_1, output_train_2, atol=1e-6), (
            "Outputs in train mode should differ due to dropout (no norm/res)."
        )


# --- Test Case 4: Activation Functions ---
@pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
@pytest.mark.slow
def test_mlp_activation_functions(activation):
    """Tests MLP with different activation functions."""
    torch.manual_seed(42)

    mlp, input_tensor = create_test_mlp(activation=activation)

    with torch.no_grad():
        output = mlp(input_tensor)

    # Basic shape test
    assert output.shape == input_tensor.shape


# --- Test Case 5: Bias Parameter ---
@pytest.mark.slow
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
@pytest.mark.parametrize("include_norm_residual_val", [True, False])
@pytest.mark.slow
def test_mlp_dropout_train_eval_modes(include_norm_residual_val):
    """
    Tests dropout behavior in train vs eval modes for MLP,
    covering both include_norm_residual=True and False.
    """
    dropout_p_test = 0.5
    torch.manual_seed(42)

    mlp, input_tensor = create_test_mlp(dropout_p=dropout_p_test, include_norm_residual=include_norm_residual_val)

    # Eval mode: Dropout should be disabled, outputs should be identical.
    mlp.eval()
    with torch.no_grad():
        output_eval_1 = mlp(input_tensor)
        output_eval_2 = mlp(input_tensor)
    assert torch.allclose(output_eval_1, output_eval_2, atol=1e-7), (
        f"Outputs in eval mode should be identical (include_norm_residual={include_norm_residual_val})"
    )

    # Train mode: Dropout should be active, outputs should differ for different forward passes.
    mlp.train()
    # It's important that multiple forward passes in train mode use different dropout masks.
    # PyTorch's dropout layer typically handles this correctly without needing to re-seed globally
    # for each call, as long as it's the same layer instance.
    with torch.no_grad():
        output_train_1 = mlp(input_tensor)
        output_train_2 = mlp(input_tensor)  # Second call to the same layer instance

    if dropout_p_test > 0:
        assert not torch.allclose(output_train_1, output_train_2, atol=1e-6), (
            f"Outputs in train mode should differ due to dropout (include_norm_residual={include_norm_residual_val})"
        )
    else:  # Should not happen with dropout_p_test = 0.5, but as a guard
        assert torch.allclose(output_train_1, output_train_2, atol=1e-7), (
            "Outputs in train mode should be identical if dropout_p is 0"
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
@pytest.mark.slow
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


# --- Test Case 8: Gradient Computation ---
@pytest.mark.parametrize("include_norm_residual_val", [True, False])
@pytest.mark.parametrize("bias_val", [True, False])  # Also test gradients with and without bias
@pytest.mark.slow
def test_mlp_gradient_computation(include_norm_residual_val, bias_val):
    """Tests if gradients are computed correctly for all trainable parameters."""
    torch.manual_seed(42)
    hidden_size_grad = 32  # Smaller hidden size for faster test
    mlp, input_tensor = create_test_mlp(
        hidden_size=hidden_size_grad,
        include_norm_residual=include_norm_residual_val,
        bias=bias_val,
        dropout_p=0.0,  # Disable dropout for deterministic gradient check
    )

    # Ensure model is in training mode for gradients
    mlp.train()
    input_tensor.requires_grad_(True)

    # Forward pass
    output = mlp(input_tensor)

    # Compute a dummy loss and backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients for all parameters that should have them
    for name, param in mlp.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN values"
            assert not torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf values"
            assert (param.grad != 0).any(), f"Gradient for {name} is all zeros (potential issue)"
        else:
            # Optional: check that non-trainable params indeed have no grad
            assert param.grad is None or (param.grad == 0).all(), f"Unexpected gradient for non-trainable param {name}"

    # Check input tensor gradient
    assert input_tensor.grad is not None, "Input tensor gradient is None"
    assert not torch.isnan(input_tensor.grad).any(), "Input tensor gradient contains NaN values"


# --- New Test Case: Test MLP with norm_type as an instance ---
@pytest.mark.slow
def test_mlp_norm_type_as_instance():
    """Tests MLP initialization and forward pass when norm_type is an nn.Module instance."""
    torch.manual_seed(42)
    hidden_size_test = 64
    custom_norm_layer = torch.nn.LayerNorm(hidden_size_test, eps=1e-6)  # Custom instance

    mlp, input_tensor = create_test_mlp(
        hidden_size=hidden_size_test,
        norm_type=custom_norm_layer,  # Pass the instance
        norm_first=True,  # Ensure norm is used
        include_norm_residual=True,
        dropout_p=0.0,  # Disable dropout for deterministic check
    )
    mlp.eval()

    assert mlp.norm is custom_norm_layer, "MLP should use the provided norm instance"
    assert mlp.norm_first is True
    assert mlp.include_norm_residual is True

    with torch.no_grad():
        output = mlp(input_tensor)

    assert output.shape == input_tensor.shape

    # Verify that the custom norm layer was indeed used by checking its parameters
    # This is a bit indirect, but ensures the path was taken.
    # If the norm layer has trainable parameters, they should be part of mlp.parameters()
    found_norm_params = False
    for name, _param in mlp.named_parameters():  # Changed param to _param
        if "norm" in name:
            found_norm_params = True
            break
    assert found_norm_params, "Custom norm layer parameters not found in MLP parameters"
