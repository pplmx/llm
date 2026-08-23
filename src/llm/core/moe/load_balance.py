"""Auxiliary load-balancing loss for MoE routing (ROADMAP 15.4 / research).

Sparse token-choice MoE (``moe.py``) routes each token to its ``top_k`` experts;
without any balancing mechanism, the gate can collapse so a few experts hoard
most tokens (load imbalance) while others go dead, hurting training stability and
capacity utilisation. The standard mitigation is the **auxiliary load-balancing
loss** introduced by Switch Transformer (Fedus et al. 2021) and ST-MoE (Zoph et
al. 2022):

    p      = softmax(gate_logits, dim=-1)          # per-token expert probs [T, E]
    idx    = argmax(p, dim=-1)                      # top-1 routed expert per token [T]
    f_i    = (1/T) sum_t 1[idx_t == i]              # fraction of tokens per expert [E]
    P_i    = (1/T) sum_t p[t, i]                    # mean dispatch prob per expert [E]
    L_aux  = aux_weight * E * sum_i f_i P_i

The product ``f_i P_i`` is large only when expert ``i`` both *receives many
tokens* AND *is strongly favoured by the gate*; summing incentivises the gate to
spread load evenly (the loss is zero-balance minimal). It is added to the total
training loss as a secondary objective with a small ``aux_weight``.

This module is a self-contained, CPU-verifiable research slice.
Wiring the loss into ``MoeLayer`` / the LM loss (add ``aux_weight * L_aux`` and
backprop) is a documented follow-up.

CPU parity invariants verified in ``tests/core/moe/test_load_balance.py``:

- balanced (uniform) routing yields a strictly smaller loss than a fully
  imbalanced (all-to-one) gate — monotone in imbalance;
- ``f`` and ``P`` each sum to ``1``;
- the loss is differentiable w.r.t. ``gate_logits`` (finite grads);
- ``aux_weight`` scales the loss linearly.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["load_balancing_loss", "mean_dispatch_probability", "routing_fractions"]


def routing_fractions(gate_logits: Tensor, num_experts: int) -> Tensor:
    """Fraction of tokens routed (top-1) to each expert. ``f``: ``[E]``, sums to 1."""
    prob = torch.softmax(gate_logits, dim=-1)
    idx = prob.argmax(dim=-1)  # [T]
    counts = torch.bincount(idx, minlength=num_experts).to(gate_logits.dtype)
    return counts / idx.numel()


def mean_dispatch_probability(gate_logits: Tensor, num_experts: int) -> Tensor:
    """Mean full-softmax dispatch probability per expert. ``P``: ``[E]``, sums to 1."""
    if gate_logits.shape[1] != num_experts:
        raise ValueError(f"gate_logits experts ({gate_logits.shape[1]}) != num_experts ({num_experts})")
    prob = torch.softmax(gate_logits, dim=-1)
    return prob.mean(dim=0)  # [E]


def load_balancing_loss(
    gate_logits: Tensor,
    num_experts: int,
    *,
    aux_weight: float = 1.0,
) -> Tensor:
    """SWITCH/ST-MoE auxiliary load-balancing loss (scalar).

    ``aux_weight * num_experts * sum_i f_i P_i``. Strictly larger for imbalanced
    routing, so adding it (with a small weight) encourages an even spread.
    """
    if aux_weight < 0:
        raise ValueError(f"aux_weight must be >= 0, got {aux_weight}")
    fracs = routing_fractions(gate_logits, num_experts)
    probs = mean_dispatch_probability(gate_logits, num_experts)
    return aux_weight * num_experts * (fracs * probs).sum()
