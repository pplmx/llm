"""Tests for QAT fake quantization (RIL TASK-218 / DEC-054)."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
import torch.optim as optim

from llm.quantization.fake_quant import FakeQuantize, FakeQuantLinear, ste_round


def test_ste_round_forward_hard_backward_identity():
    x = torch.tensor([1.2, -2.6, 0.4, 3.5], requires_grad=True)
    out = ste_round(x)
    assert torch.equal(out, x.round())
    out.sum().backward()
    # The STE passes the gradient through unchanged (1 for every element).
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x))


def test_fake_quantize_matches_symmetric_grid_with_explicit_scale():
    fq = FakeQuantize(8)
    x = torch.tensor([-200.0, -130.0, 0.0, 50.0, 127.0, 400.0])
    q = fq(x, scale=torch.tensor(1.0))
    expected = x.round().clamp(-127, 127)
    assert torch.equal(q, expected)


def test_fake_quantize_refuses_invalid_bits():
    for bad in (3, 6, 16):
        try:
            FakeQuantize(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"FakeQuantize({bad}) should have raised")


def _hard_per_channel(w, qmax):
    """Reference: per-channel symmetric fake quant (dynamic scale), no STE."""
    amax = w.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    return (w / scale).round().clamp(-qmax, qmax) * scale


def test_fake_quant_linear_forward_matches_hard_quant():
    torch.manual_seed(0)
    qmax = 127.0
    layer = FakeQuantLinear(32, 16, bits=8, quant_activation=False)
    w = torch.randn(16, 32) * 3
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        layer.weight.copy_(w)
        layer.bias.zero_()
    y = layer(x)
    wq = _hard_per_channel(w, qmax)
    expected = functional.linear(x.float(), wq, None)
    torch.testing.assert_close(y, expected, atol=1e-5, rtol=1e-5)


def test_fake_quant_linear_trains_and_reduces_loss():
    """QAT: fake-quantized linear learns (loss decreases) via STE gradients."""
    torch.manual_seed(3)
    layer = FakeQuantLinear(16, 4, bits=8, quant_activation=True)
    target = torch.randn(4, 16) * 2
    x = torch.randn(64, 16)
    y = x @ target.t() + torch.randn(4) * 0.0
    opt = optim.SGD(layer.parameters(), lr=0.05)
    criterion = torch.nn.functional.mse_loss
    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = criterion(layer(x), y)
        loss.backward()
        grads_finite = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in layer.parameters())
        assert grads_finite, "fake-quant STE must give finite gradients on every parameter"
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], "QAT fake-quant training should reduce the loss"


def test_fake_quant_roundtrip_error_is_bounded():
    """Reconstruction error is bounded by the dynamic scale (<= qmax/2 grid)."""
    torch.manual_seed(1)
    x = torch.randn(1, 64) * 10
    fq = FakeQuantize(8)
    q = fq(x)
    err = (q - x).abs().max().item()
    # With dynamic per-tensor scale = absmax/qmax, the max grid step is
    # absmax_block/qmax ~ (10*qmax-ish)/qmax = 10, half-step ~5; assert a
    # tighter empirical bound than random round-trip (signal it's fine).
    assert err < 1.0, f"round-trip error {err} should be small for a 10-range input"
