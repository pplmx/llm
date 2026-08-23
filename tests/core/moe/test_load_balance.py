"""Tests for the auxiliary load-balancing loss slice (ROADMAP 15.4)."""

from __future__ import annotations

import pytest
import torch

from llm.core.moe.load_balance import load_balancing_loss, mean_dispatch_probability, routing_fractions


def test_balanced_lower_than_imbalanced():
    """A near-uniform gate (balanced routing) yields a strictly smaller auxiliary
    loss than a fully imbalanced all-to-one gate."""
    n_tok, n_exp = 80, 4
    balanced = torch.randn(n_tok, n_exp) * 0.1  # near-uniform softmax
    imbalanced = torch.zeros(n_tok, n_exp)
    imbalanced[:, 0] = 5.0
    lb_bal = load_balancing_loss(balanced, n_exp)
    lb_imb = load_balancing_loss(imbalanced, n_exp)
    assert lb_bal < lb_imb
    assert torch.isfinite(lb_bal)


def test_fractions_and_probs_sum_to_one():
    torch.manual_seed(0)
    gate = torch.randn(64, 4)
    fracs = routing_fractions(gate, 4)
    probs = mean_dispatch_probability(gate, 4)
    assert torch.allclose(fracs.sum(), torch.ones(()), atol=1e-5)
    assert torch.allclose(probs.sum(), torch.ones(()), atol=1e-5)
    assert fracs.shape == (4,)
    assert probs.shape == (4,)


def test_gradient_flow_is_finite():
    torch.manual_seed(1)
    gate = torch.randn(40, 3, requires_grad=True)
    loss = load_balancing_loss(gate, 3)
    loss.backward()
    assert gate.grad is not None
    assert bool(torch.isfinite(gate.grad).all())


def test_aux_weight_scales_linearly():
    torch.manual_seed(2)
    gate = torch.randn(50, 3)
    base = load_balancing_loss(gate, 3)
    doubled = load_balancing_loss(gate, 3, aux_weight=2.0)
    assert torch.allclose(doubled, 2.0 * base, atol=1e-6)
    # Zero weight is a no-op.
    assert float(load_balancing_loss(gate, 3, aux_weight=0.0)) == 0.0


def test_negative_aux_weight_rejected():
    gate = torch.randn(20, 3)
    with pytest.raises(ValueError, match="aux_weight"):
        load_balancing_loss(gate, 3, aux_weight=-0.1)
