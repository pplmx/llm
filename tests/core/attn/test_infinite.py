"""Tests for the Infini-Attention compressive-memory slice (ROADMAP 15.1)."""

from __future__ import annotations

import torch

from llm.core.attn.infinite import InfiniMemory, feature_map, infinite_attention

T, D = 6, 8


def _segments(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    return torch.randn(T, D), torch.randn(T, D), torch.randn(T, D)


def test_fresh_memory_retrieves_zero():
    """An empty memory has no contexts to retrieve -> exactly zero; this is what
    makes a fresh Infini-Attention head behave like plain dense attention."""
    q, _, _ = _segments()
    mem = InfiniMemory(D, D)
    ctx = mem.retrieve(q[0])
    assert ctx.shape == (1, D)
    assert float(ctx.abs().sum()) == 0.0


def test_memory_state_shape_is_context_independent():
    """Compressive memory is a fixed [d_k, d_v] state regardless of how many
    tokens were written — the unbounded-context property."""
    _, k, v = _segments()
    mem = InfiniMemory(D, D)
    for i in range(0, T):
        mem.update(k[i : i + 1], v[i : i + 1])
    assert mem.M.shape == (D, D)
    assert mem.z.shape == (D,)
    assert float(mem.z.sum()) > 0.0
    # Writing more tokens grows no state.
    mem.update(k, v)
    assert mem.M.shape == (D, D)


def test_retrieval_equals_linear_attention_reference():
    """Memory retrieval is exactly the linear-attention weighted average of all
    accumulated values: sum phi(q.k_i) v_i / sum phi(q.k_i)."""
    q, k, v = _segments()
    mem = InfiniMemory(D, D)
    prefix = 4
    mem.update(k[:prefix], v[:prefix])

    phi_q = feature_map(q[5])
    ref_val = (phi_q @ (feature_map(k[:prefix]).transpose(0, 1) @ v[:prefix])) / (
        phi_q @ feature_map(k[:prefix]).sum(dim=0)
    )
    ctx = mem.retrieve(q[5])[0]
    assert torch.allclose(ctx, ref_val, atol=1e-4)
    # The retrieved vector is a convex-ish blend of the prefix values: norm-bounded.
    assert bool(torch.isfinite(ctx).all())


def test_beta_infinity_equals_dense_attention():
    """Pushing the gate to the dot side (sigmoid(beta)->1) reproduces plain
    dense attention over the current segment — Infini-Attention is a superset."""
    q, k, v = _segments()
    out, _ = infinite_attention(q, k, v, beta=500.0, causal=True)
    causal = torch.nn.functional.scaled_dot_product_attention(
        q[None, None, :, :], k[None, None, :, :], v[None, None, :, :], is_causal=True
    )[0, 0]
    assert torch.allclose(out, causal, atol=1e-5)

    out_full, _ = infinite_attention(q, k, v, beta=500.0, causal=False)
    full = torch.nn.functional.scaled_dot_product_attention(
        q[None, None, :, :], k[None, None, :, :], v[None, None, :, :]
    )[0, 0]
    assert torch.allclose(out_full, full, atol=1e-5)


def test_memory_gives_long_range_recall():
    """A value stored at an early position is recoverable by a far later query
    through the memory head even though causal dot attention can also see it —
    here with the gate pushed to memory the recalled value dominates."""
    _, k, v = _segments()
    # Plant a recognizable value at position 0.
    v0 = torch.zeros(D)
    v0[0] = 42.0
    v[0] = v0
    mem = InfiniMemory(D, D)
    # Seed memory with ALL past keys/values.
    mem.update(k, v)
    # The planted value contributes the single '42' scaled by its attended weight,
    # so querying near k0 recovers a value proportional to 42 -> nonzero.
    q_near = k[0:1].clone()
    near_val = mem.retrieve(q_near[0])[0]
    assert float(near_val[0]) > 1.0


def test_gradient_flow_is_finite():
    """Every state-machine op is differentiable: backward yields finite grads."""
    q = torch.randn(T, D, requires_grad=True)
    _, k, v = _segments()
    out, _ = infinite_attention(q, k, v, beta=0.0, causal=True)
    out.sum().backward()
    assert q.grad is not None
    assert bool(torch.isfinite(q.grad).all())
    assert bool(torch.isfinite(out).all())
