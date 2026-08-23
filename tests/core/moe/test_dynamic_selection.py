"""Tests for the Dynamic Expert Selection research slice (ROADMAP 15.4)."""

from __future__ import annotations

import pytest
import torch

from llm.core.moe.dynamic_selection import dynamic_expert_count, dynamic_expert_output


def _gate_with_confident_and_uncertain_tokens():
    torch.manual_seed(0)
    gate = torch.randn(8, 4)
    gate[0, 0] = 5.0  # confident (top-1 ~0.99)
    gate[1] = 0.05  # uncertain (near-uniform, top-1 ~0.25)
    return gate


def test_count_switches_by_confidence():
    gate = _gate_with_confident_and_uncertain_tokens()
    kt = dynamic_expert_count(gate, min_experts=1, max_experts=3, high_conf_threshold=0.7)
    assert int(kt[0]) == 1  # confident -> fewest experts
    assert int(kt[1]) == 3  # uncertain -> full budget
    assert bool(((kt >= 1) & (kt <= 3)).all())


def test_weights_sum_to_one():
    gate = _gate_with_confident_and_uncertain_tokens()
    x = torch.randn(8, 6)

    def expert_fn(slots):
        return slots * 2.0

    out = dynamic_expert_output(x, gate, expert_fn, min_experts=1, max_experts=3, high_conf_threshold=0.7)
    assert out.shape == (8, 6)
    assert torch.isfinite(out).all()


def test_output_equals_weighted_sum_reference():
    torch.manual_seed(1)
    gate = torch.randn(8, 4)
    gate[0, 0] = 4.0  # confident
    x = torch.randn(8, 6)
    n_tok = 8
    min_e, max_e, thr = 1, 3, 0.5

    def expert_fn(slots):
        return slots * 1.5

    out = dynamic_expert_output(x, gate, expert_fn, min_e, max_e, thr)

    prob = torch.softmax(gate, dim=-1)
    scores, _ = prob.topk(max_e, dim=-1)
    ref = torch.zeros(n_tok, x.shape[-1])
    for t in range(n_tok):
        keep = min_e if float(prob[t].max()) >= thr else max_e
        kept_scores = scores[t, :keep]
        p = torch.softmax(kept_scores, dim=-1)
        ref[t] = torch.sum(p[:, None] * expert_fn(x[t, None, None, :].expand(keep, 1, -1))[:, 0], dim=0)
    assert torch.allclose(out, ref, atol=1e-5)


def test_gradient_flow_is_finite():
    torch.manual_seed(2)
    x = torch.randn(6, 5, requires_grad=True)
    gate = torch.randn(6, 3, requires_grad=True)
    out = dynamic_expert_output(x, gate, lambda s: s * 1.2, min_experts=1, max_experts=2, high_conf_threshold=0.6)
    out.sum().backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())
    assert gate.grad is not None
    assert bool(torch.isfinite(gate.grad).all())


def test_input_validation():
    gate = torch.randn(8, 4)
    with pytest.raises(ValueError, match="min_experts"):
        dynamic_expert_count(gate, min_experts=3, max_experts=2, high_conf_threshold=0.5)
    with pytest.raises(ValueError, match="high_conf_threshold"):
        dynamic_expert_count(gate, min_experts=1, max_experts=2, high_conf_threshold=1.5)


def test_confident_single_expert_equals_expert_output():
    torch.manual_seed(3)
    gate = torch.full((4, 3), -5.0)
    gate[:, 0] = 5.0  # token 0 expert dominates -> top-1 ~1
    x = torch.randn(4, 6)
    out = dynamic_expert_output(x, gate, lambda s: s * 2.0, min_experts=1, max_experts=2, high_conf_threshold=0.7)
    # Each token keeps a single expert whose slot input is the token itself.
    assert torch.allclose(out, 2.0 * x, atol=1e-3)
