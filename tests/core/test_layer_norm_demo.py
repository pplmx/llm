"""
Layer Normalization Demo Tests

Demonstrates and validates Layer Normalization implementation against PyTorch built-in.
"""

import numpy as np
import torch
import torch.nn as nn

from llm.core.layer_norm import LayerNorm, layer_norm_numpy


def test_layer_norm_implementations():
    """Test custom LayerNorm against PyTorch built-in implementation."""
    # Parameters
    batch_size = 2
    seq_length = 3
    hidden_size = 4
    normalized_shape = (hidden_size,)

    # Create input data
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(batch_size, seq_length, hidden_size)
    x_numpy = x.detach().numpy().copy()

    # Initialize LayerNorm modules
    torch_layer_norm = nn.LayerNorm(normalized_shape=normalized_shape, eps=1e-5, elementwise_affine=True)
    custom_layer_norm = LayerNorm(normalized_shape=normalized_shape, eps=1e-5, elementwise_affine=True)
    custom_layer_norm_no_affine = LayerNorm(normalized_shape=normalized_shape, eps=1e-5, elementwise_affine=False)

    # Forward pass
    torch_output = torch_layer_norm(x)

    # Copy weights for fair comparison
    custom_layer_norm.weight.data = torch_layer_norm.weight.data.clone()
    custom_layer_norm.bias.data = torch_layer_norm.bias.data.clone()
    custom_output = custom_layer_norm(x)
    custom_output_no_affine = custom_layer_norm_no_affine(x)

    # NumPy implementation
    gamma_numpy = torch_layer_norm.weight.data.detach().numpy()
    beta_numpy = torch_layer_norm.bias.data.detach().numpy()
    numpy_output = layer_norm_numpy(x_numpy, gamma=gamma_numpy, beta=beta_numpy, eps=1e-5)
    numpy_output_no_affine = layer_norm_numpy(x_numpy, gamma=None, beta=None, eps=1e-5)

    # Assertions
    assert torch.allclose(torch_output, custom_output, atol=1e-5), (
        "Custom PyTorch implementation does not match PyTorch built-in"
    )
    assert np.allclose(custom_output.detach().numpy(), numpy_output, atol=1e-5), (
        "NumPy implementation does not match PyTorch"
    )
    assert np.allclose(custom_output_no_affine.detach().numpy(), numpy_output_no_affine, atol=1e-5), (
        "No-affine versions do not match"
    )

    # Check output shapes
    assert torch_output.shape == (batch_size, seq_length, hidden_size)
    assert custom_output.shape == (batch_size, seq_length, hidden_size)


def test_layer_norm_statistics():
    """Test that LayerNorm produces expected statistical properties."""
    batch_size = 2
    seq_length = 3
    hidden_size = 4
    normalized_shape = (hidden_size,)

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_length, hidden_size)

    ln = LayerNorm(normalized_shape=normalized_shape, elementwise_affine=False)
    output = ln(x)

    # For no-affine LayerNorm, output should have mean ~0 and std ~1
    means = output.mean(dim=-1)
    stds = output.std(dim=-1, unbiased=False)

    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5), "LayerNorm output mean should be close to 0"
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-4), "LayerNorm output std should be close to 1"
