"""Tests for the Soft MoE research slice (ROADMAP 15.4)."""

from __future__ import annotations

import pytest
import torch

from llm.core.moe.soft_moe import dispatch_weights, soft_moe_output


def test_every_slot_sums_to_one():
    """Dispatch weights are a softmax over tokens per slot -> perfect slot
    balance by construction (each slot carries total mass 1)."""
    torch.manual_seed(0)
    slot_logits = torch.randn(8, 6)  # E*S=8, T=6
    d = dispatch_weights(slot_logits, num_experts=4, num_slots=2)
    assert d.shape == (4, 2, 6)
    assert torch.allclose(d.sum(dim=-1), torch.ones(4, 2), atol=1e-5)
    assert torch.isfinite(d).all()


def test_output_equals_weighted_sum_reference():
    """y_t = sum_{e,s} D[e,s,t] * E_s(slot input) matches an explicit loop."""
    torch.manual_seed(1)
    d, n_tok, n_exp, n_slot = 6, 8, 4, 2
    x = torch.randn(n_tok, d)
    slot_logits = torch.randn(n_exp * n_slot, n_tok)
    d_weights = dispatch_weights(slot_logits, n_exp, n_slot)

    def expert_fn(sel):
        return sel * 2.0

    y = soft_moe_output(x, expert_fn, slot_logits, n_exp, n_slot)

    ref = torch.zeros(n_tok, d)
    for ex in range(n_exp):
        for sl in range(n_slot):
            slot_in = torch.sum(d_weights[ex, sl, :, None] * x, dim=0)  # [d]
            out = expert_fn(slot_in.unsqueeze(0))[0]  # [d]
            ref = ref + d_weights[ex, sl, :, None] * out
    assert torch.allclose(y, ref, atol=1e-5)


def test_gradient_flow_is_finite():
    torch.manual_seed(2)
    d, n_tok, n_exp, n_slot = 5, 6, 3, 2
    x = torch.randn(n_tok, d, requires_grad=True)
    slot_logits = torch.randn(n_exp * n_slot, n_tok, requires_grad=True)

    y = soft_moe_output(x, lambda s: s * 1.5, slot_logits, n_exp, n_slot)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())
    assert slot_logits.grad is not None
    assert bool(torch.isfinite(slot_logits.grad).all())


def test_input_validation():
    slot_logits = torch.randn(6, 5)
    with pytest.raises(ValueError, match="num_experts"):
        dispatch_weights(slot_logits, num_experts=0, num_slots=2)
    with pytest.raises(ValueError, match="slot_logits first dim"):
        dispatch_weights(slot_logits, num_experts=2, num_slots=2)  # 4 != 6


def test_slot_selects_single_token_in_extreme_limit():
    """As one token's score dominates a slot, the slot input collapses toward
    that single token, so the output for it recovers ~ the expert on that token."""
    torch.manual_seed(3)
    d, n_tok = 5, 4
    x = torch.randn(n_tok, d)
    slot_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])  # token 0 dominates
    out = soft_moe_output(x, lambda s: s, slot_logits, num_experts=1, num_slots=1)
    # With weight ~1 on token 0, its output ~= x[0]; other tokens get ~0.
    assert torch.allclose(out[0], x[0], atol=1e-3)
    assert torch.allclose(out[1:], torch.zeros(n_tok - 1, d), atol=1e-3)
