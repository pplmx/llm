"""RMSNorm implementation vs NumPy reference."""

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


def test_rms_norm_fp16_large_activations_no_collapse():
    """Half-precision RMSNorm must not collapse to zero for large activations
    (RIL ISS-384). ``pow(2)`` in fp16 overflows at |x| > ~256 (fp16 max
    65504), so a naive ``x.pow(2).mean()`` in host dtype produced an inf
    mean-square and the normalized output silently collapsed to all-zeros.
    The mean-square must be accumulated in fp32 (cast back for the RMS),
    as standard RMSNorm implementations do."""
    # hidden=4096: the pow(2) sum is ~3.7e8 for |x|=300 — far past fp16 max.
    rms_norm = RMSNorm(normalized_shape=4096, eps=1e-6, elementwise_affine=False, dtype=torch.float16)
    rms_norm.eval()

    x = torch.full((2, 12, 4096), 300.0, dtype=torch.float16)
    with torch.no_grad():
        out = rms_norm(x)

    # 300 / sqrt(300^2 + eps) = 1.0 — must NOT collapse to 0 or NaN/inf.
    assert torch.isfinite(out).all(), "fp16 RMSNorm produced non-finite output"
    assert not torch.all(out == 0), "fp16 RMSNorm collapsed to all-zeros (mean-square overflowed)"
    assert torch.allclose(out.float(), torch.ones_like(out), atol=1e-2), (
        "fp16 RMSNorm output should be ~1.0 for constant |x|=300 input"
    )


def test_rms_norm_fp16_matches_fp32_reference():
    """Half-precision RMSNorm must agree with the fp32 computation on
    moderate-magnitude inputs (RIL ISS-384). The fp32 accumulation path is
    the reference; the fp16 path must produce near-identical output."""
    torch.manual_seed(7)
    x = torch.randn(2, 8, 1024, dtype=torch.float32) * 3.0

    rn16 = RMSNorm(normalized_shape=1024, eps=1e-6, elementwise_affine=True, dtype=torch.float16).eval()
    rn32 = RMSNorm(normalized_shape=1024, eps=1e-6, elementwise_affine=True, dtype=torch.float32).eval()
    rn32.weight.data.copy_(rn16.weight.data.float())

    with torch.no_grad():
        out16 = rn16(x.half()).float()
        out32 = rn32(x)

    assert torch.allclose(out16, out32, atol=1e-2, rtol=1e-2), (
        f"fp16 RMSNorm diverged from fp32 reference: max abs diff {(out16 - out32).abs().max().item():.4f}"
    )
