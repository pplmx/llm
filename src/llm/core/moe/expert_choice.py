"""Expert Choice Routing research slice (ROADMAP 15.4 / research).

Standard sparse MoE (token-choice) has each *token* route itself to its ``top_k``
experts (see ``moe.py``). Expert Choice Routing (Zhou et al., "Mixture-of-Experts
with Expert Choice Routing", 2022) inverts this: each *expert* selects its own
``top_k`` tokens from the whole batch. The decisive property it buys is **perfect
load balance by construction** — every expert gets exactly ``k`` tokens, so no
expert is starved (dead) or overloaded, and no token is dropped. A token may be
selected by any number of experts (including all or none).

This module implements the routing + weighting + combine as a self-contained,
CPU-verifiable research slice (same spirit as the ``core/attn`` long-context
builders). Wiring it as an alternate ``MoeLayer.forward`` routing mode is a
documented follow-up.

CPU parity invariants verified in ``tests/core/moe/test_expert_choice.py``:

- each expert's selected tokens are exactly its ``top_k`` by gate score;
- every expert receives exactly ``k`` tokens (guaranteed load balance);
- the combined output equals the explicit weighted-sum reference over the
  selected (expert, token) pairs;
- the routing is differentiable (backward yields finite grads).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = [
    "expert_choice_assignment",
    "expert_choice_output",
    "expert_choice_weights",
]


def expert_choice_assignment(gate_logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Select, for every expert, its ``top_k`` tokens by gate score.

    Args:
        gate_logits: ``[num_tokens, num_experts]`` — expert affinities per token.
        k: Number of tokens each expert selects (``1 <= k <= num_tokens``).

    Returns:
        ``(tokens_per_expert [E, k], scores_per_expert [E, k])`` where
        ``tokens_per_expert[e]`` are the global token indices expert ``e`` picks
        and ``scores_per_expert[e]`` their raw gate scores.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    num_tokens, _ = gate_logits.shape
    if k > num_tokens:
        raise ValueError(f"k ({k}) cannot exceed num_tokens ({num_tokens})")
    # [E, T] so topk over the token axis per expert.
    per_expert = gate_logits.transpose(0, 1)  # [E, T]
    scores, tokens = torch.topk(per_expert, k, dim=1)  # [E, k], [E, k]
    return tokens, scores


def expert_choice_weights(scores_per_expert: Tensor) -> Tensor:
    """Normalize each expert's selected-token scores into routing weights.

    Per-expert softmax over its ``k`` chosen tokens (the paper's normalization):
    weights ``[E, k]``, ``sum_k == 1`` for every expert.
    """
    return torch.softmax(scores_per_expert, dim=-1)


def expert_choice_output(
    x: Tensor,
    gate_logits: Tensor,
    expert_fn: Callable[[Tensor], Tensor],
    k: int,
) -> Tensor:
    """Route ``x`` through Expert Choice and combine each expert's output.

    Args:
        x: ``[num_tokens, d]`` input tokens.
        gate_logits: ``[num_tokens, num_experts]`` gate scores.
        expert_fn: Callable expert block, ``(tokens [n, d]) -> (out [n, d])``
            applied to the ``n`` tokens each expert selected.
        k: Tokens selected per expert.

    Returns:
        Combined output ``[num_tokens, d]``; token ``t`` accumulates
        ``sum over experts e that selected t`` of ``w[e, t] * expert_e(x[t])``.
    """
    _, num_experts = gate_logits.shape
    tokens_per_expert, scores_per_expert = expert_choice_assignment(gate_logits, k)
    weights = expert_choice_weights(scores_per_expert)  # [E, k]

    out = torch.zeros_like(x)
    x = x.contiguous()
    for e in range(num_experts):
        tok_ids = tokens_per_expert[e]  # [k]
        selected = x.index_select(0, tok_ids)  # [k, d]
        expert_out = expert_fn(selected)  # [k, d]
        weighted = expert_out * weights[e].unsqueeze(-1)  # [k, d]
        # Scatter-add back to the token rows.
        out = out.index_add(0, tok_ids, weighted)
    return out
