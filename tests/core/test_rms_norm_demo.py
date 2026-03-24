"""
RMS Normalization Demo Tests

Demonstrates and validates RMS Normalization implementation.
"""

import numpy as np
import torch

from llm.core.rms_norm import RMSNorm, rms_norm_numpy


def test_rms_norm_implementations():
    """Test custom RMSNorm implementation against NumPy reference."""
    # Parameters
    batch_size = 2
    seq_length = 3
    hidden_size = 4
    normalized_shape = (hidden_size,)
    eps = 1e-6

    # Create input data
    torch.manual_seed(43)
    np.random.seed(43)
    x = torch.randn(batch_size, seq_length, hidden_size) * 2
    x_numpy = x.detach().numpy().copy()

    # Initialize RMSNorm modules
    custom_rms_norm = RMSNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=True)
    custom_rms_norm_no_affine = RMSNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=False)

    # Forward pass
    custom_output = custom_rms_norm(x)
    custom_output_no_affine = custom_rms_norm_no_affine(x)

    # NumPy implementation
    gamma_numpy = custom_rms_norm.weight.data.detach().numpy().copy()
    numpy_output = rms_norm_numpy(x_numpy, gamma=gamma_numpy, eps=eps)
    numpy_output_no_affine = rms_norm_numpy(x_numpy, gamma=None, eps=eps)

    # Assertions
    assert np.allclose(custom_output.detach().numpy(), numpy_output, atol=1e-5), (
        "RMSNorm (affine) NumPy implementation does not match PyTorch"
    )
    assert np.allclose(custom_output_no_affine.detach().numpy(), numpy_output_no_affine, atol=1e-5), (
        "RMSNorm (no-affine) NumPy implementation does not match PyTorch"
    )

    # Check output shapes
    assert custom_output.shape == (batch_size, seq_length, hidden_size)
    assert custom_output_no_affine.shape == (batch_size, seq_length, hidden_size)


def test_rms_norm_rms_property():
    """Test that RMSNorm produces expected RMS property."""
    batch_size = 2
    seq_length = 3
    hidden_size = 4
    normalized_shape = (hidden_size,)

    torch.manual_seed(43)
    x = torch.randn(batch_size, seq_length, hidden_size) * 2

    rms_norm = RMSNorm(normalized_shape=normalized_shape, elementwise_affine=False)
    output = rms_norm(x)

    # For no-affine RMSNorm, output RMS should be close to 1.0
    output_rms = torch.sqrt(torch.mean(output.pow(2), dim=-1))

    assert torch.allclose(output_rms, torch.ones_like(output_rms), atol=1e-5), "RMSNorm output RMS should be close to 1"
