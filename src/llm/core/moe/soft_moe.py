"""Soft MoE research slice (ROADMAP 15.4 / research).

Soft MoE (Puigcerver et al., "From Sparse to Soft Mixtures of Experts", 2023)
replaces the discrete top-``k`` routing of dense/sparse MoE with a
deterministic, fully differentiable *slot* mechanism. Each expert owns a few
slots; every slot computes a normalized soft assignment ``D`` over all tokens
(softmax over tokens, per slot), so:

- the slot input is the weighted average ``sum_t D[s,t] x_t`` (any token may
  flow into any slot — no token is dropped, no hard argmax/topk);
- the expert runs on each slot input;
- the expert output is dispatched back to every token as a convex combination
  ``y_t += sum_s D[s,t] * E_s(slot_s)``.

Tying dispatch and combine weights to the same ``D`` is the model used here.
Key property it buys vs discrete routing: **perfect slot balance by
construction** — ``sum_t D[e,s,t] == 1`` for every slot, so no expert/slot is
starved or overwhelmed and the routing is smooth (differentiable), avoiding the
load-balancing discontinuities of hard top-k.

This module is a self-contained, CPU-verifiable research slice in the same
spirit as ``moe/expert_choice.py``. Wiring it as an alternate MoE forward
(learnable slot-logits projection + ``MoeLayer`` integration) is a documented
follow-up.

CPU parity invariants verified in ``tests/core/moe/test_soft_moe.py``:

- every slot's dispatch weights sum to exactly ``1`` (structural balance);
- the output equals the explicit per-slot weighted-sum reference;
- the whole path is differentiable (backward yields finite grads).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = ["dispatch_weights", "soft_moe_output"]


def dispatch_weights(slot_logits: Tensor, num_experts: int, num_slots: int) -> Tensor:
    """Per-slot soft assignment over tokens.

    Args:
        slot_logits: ``[E*S, T]`` unnormalized slot affinities (one row per
            (expert, slot), one column per token).
        num_experts: Number of experts (``E``).
        num_slots: Number of slots per expert (``S``).

    Returns:
        ``D [E, S, T]``; ``softmax over tokens`` for each slot, so every slot's
        row sums to ``1`` — perfect slot balance by construction.
    """
    if num_experts < 1 or num_slots < 1:
        raise ValueError(f"num_experts/num_slots must be >= 1, got {num_experts}/{num_slots}")
    e_s, t = slot_logits.shape
    if e_s != num_experts * num_slots:
        raise ValueError(f"slot_logits first dim ({e_s}) != num_experts*num_slots ({num_experts * num_slots})")
    soft = torch.softmax(slot_logits, dim=-1)  # [E*S, T]
    return soft.view(num_experts, num_slots, t)


def soft_moe_output(
    x: Tensor,
    expert_fn: Callable[[Tensor], Tensor],
    slot_logits: Tensor,
    num_experts: int,
    num_slots: int,
) -> Tensor:
    """Route ``x`` through Soft MoE and return the combined output ``[T, d]``.

    Each slot forms a token-weighted input and its expert output is returned to
    every token via the same dispatch weights:

        y_t = sum_{e,s} D[e,s,t] * expert_fn( sum_{t'} D[e,s,t'] x_{t'} )[e,s]
    """
    d_weights = dispatch_weights(slot_logits, num_experts, num_slots)  # [E, S, T]
    e_s = num_experts * num_slots

    # Slot inputs: weighted average of tokens per (expert, slot). [E, S, d].
    slot_in = torch.einsum("est,td->esd", d_weights, x)
    expert_out = expert_fn(slot_in.reshape(e_s, x.shape[-1]))  # [E*S, d]
    expert_out = expert_out.view(num_experts, num_slots, x.shape[-1])  # [E, S, d]

    # Dispatch back: y_t = sum_{e,s} D[e,s,t] * expert_out[e,s]. [T, d].
    return torch.einsum("est,esd->td", d_weights, expert_out)
