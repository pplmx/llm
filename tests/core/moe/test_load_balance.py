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


def test_perfectly_uniform_routing_hits_the_theoretical_minimum():
    """Uniform routing must give EXACTLY 1.0, not merely "less than imbalanced".

    ``load_balancing_loss = aux_weight * num_experts * sum(f_i * P_i)``. With
    uniform dispatch ``f_i = P_i = 1/E``, so the sum is ``E * (1/E^2) = 1/E`` and
    the product is exactly 1.0 for ANY expert count. The existing comparison
    test only checks balanced < imbalanced, which holds even if a constant
    factor is wrong — pinning the absolute value catches that.
    """
    expected = 1.0
    for num_experts in (2, 4, 8):
        # Identical logits -> uniform softmax -> perfectly balanced dispatch.
        gate = torch.zeros(64, num_experts)
        value = float(load_balancing_loss(gate, num_experts))
        assert value == pytest.approx(expected, abs=1e-6), (
            f"{num_experts} experts: uniform routing gave {value}, expected exactly {expected}"
        )


def test_total_collapse_hits_the_maximum():
    """Sending every token to one expert must give exactly ``num_experts``.

    At full collapse ``sum(f_i * P_i) = 1``, so the result is ``num_experts`` —
    the other end of the range, and equally independent of any constant factor.
    """
    for num_experts in (2, 4, 8):
        collapsed = torch.full((64, num_experts), -20.0)
        collapsed[:, 0] = 20.0
        value = float(load_balancing_loss(collapsed, num_experts))
        assert value == pytest.approx(float(num_experts), rel=1e-4), (
            f"{num_experts} experts: full collapse gave {value}, expected {num_experts}"
        )
