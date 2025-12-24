import pytest
import torch.nn as nn

from llm.utils.common import get_activation_layer


@pytest.mark.quick
def test_get_activation_layer_supported_activations():
    """Test that get_activation_layer returns the correct activation class for supported names."""
    # Test cases: (activation_name, expected_class)
    test_cases = [
        ("gelu", nn.GELU),
        ("GELU", nn.GELU),  # Test case-insensitivity
        ("relu", nn.ReLU),
        ("ReLU", nn.ReLU),
        ("silu", nn.SiLU),
        ("SiLU", nn.SiLU),
        ("swish", nn.SiLU),  # swish is an alias for SiLU
        ("SWISH", nn.SiLU),
        ("glu", nn.GLU),  # GLU returns a lambda that instantiates GLU
        ("GLU", nn.GLU),
        ("leaky_relu", nn.LeakyReLU),
        ("LEAKY_RELU", nn.LeakyReLU),
    ]

    for name, expected_class in test_cases:
        activation_class = get_activation_layer(name)
        # For GLU, the function returns a lambda that creates an instance.
        # We need to check if the returned callable creates an instance of the expected class.
        if expected_class == nn.GLU:
            # Call the lambda to get an instance, then check its type
            instance = activation_class()  # GLU requires dim, but for type check, we can pass a dummy
            assert isinstance(instance, expected_class)
        else:
            assert activation_class == expected_class


@pytest.mark.quick
def test_get_activation_layer_unsupported_activation():
    """Test that get_activation_layer raises ValueError for unsupported activation names."""
    with pytest.raises(
        ValueError, match="Unsupported activation function: unknown_activation"
    ):  # Updated error message
        get_activation_layer("unknown_activation")


@pytest.mark.quick
def test_get_activation_layer_case_insensitivity():
    """Test that get_activation_layer is case-insensitive."""
    assert get_activation_layer("GeLu") == nn.GELU
    assert get_activation_layer("ReLU") == nn.ReLU
    assert get_activation_layer("SiLu") == nn.SiLU
    assert get_activation_layer("SwIsH") == nn.SiLU
    assert get_activation_layer("GlU") == nn.GLU
    assert get_activation_layer("LeAkY_ReLu") == nn.LeakyReLU
