"""Dynamic Expert Selection research slice (ROADMAP 15.4 / research).

Standard sparse MoE routes every token to a fixed ``top_k`` experts. **Dynamic
expert selection** adapts the per-token expert count to routing confidence:
an "easy" token whose top-1 gate probability is high needs only a few experts,
while an uncertain token gets more capacity. This saves compute on confident
tokens and gives harder tokens more heads, without any hard-dynamic (data
dependent) graph.

Concretely this slice implements a two-tier adaptive scheme:

- ``p = softmax(gate_logits, dim=-1)`` (per-token expert probabilities);
- ``k_t = min_experts`` when ``max(p) >= high_conf_threshold`` (confident ->
  small), else ``max_experts`` (uncertain -> full budget);
- the ``max_experts`` top experts are ranked once and, per token, the kept set
  is the first ``k_t`` of them, renormalized to a valid distribution.

Fully differentiable (softmax everywhere, no argmax in the weight path) and
CPU-verifiable. Self-contained research slice; wiring the adaptive count into
``MoeLayer.forward`` (variable-capacity dispatch) is a documented follow-up.

CPU parity invariants verified in ``tests/core/moe/test_dynamic_selection.py``:

- confident tokens (top-1 >= threshold) use exactly ``min_experts``, uncertain
  ones exactly ``max_experts``; the count stays within ``[min, max]``;
- per-token weights sum to ``1`` (a valid distribution over the kept experts);
- the output equals the explicit renormalized weighted-sum reference;
- the path is differentiable (finite grads on ``gate_logits`` and ``x``).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = [
    "dynamic_expert_count",
    "dynamic_expert_output",
    "top1_confidence",
]


def top1_confidence(gate_logits: Tensor) -> Tensor:
    """Per-token top-1 softmax probability. ``[T]`` in ``[0, 1]``."""
    return torch.softmax(gate_logits, dim=-1).max(dim=-1).values


def dynamic_expert_count(
    gate_logits: Tensor,
    min_experts: int,
    max_experts: int,
    high_conf_threshold: float,
) -> Tensor:
    """Per-token adaptive expert count ``[T]``.

    ``min_experts`` when the token is confident (top-1 prob >= threshold),
    ``max_experts`` otherwise; both must satisfy ``1 <= min <= max`` and the
    threshold ``0 <= high_conf_threshold <= 1``.
    """
    if not 1 <= min_experts <= max_experts:
        raise ValueError(f"require 1 <= min_experts <= max_experts, got {min_experts}/{max_experts}")
    if not 0.0 <= high_conf_threshold <= 1.0:
        raise ValueError(f"high_conf_threshold must be in [0,1], got {high_conf_threshold}")
    conf = top1_confidence(gate_logits)
    return torch.where(conf >= high_conf_threshold, min_experts, max_experts)


def dynamic_expert_output(
    x: Tensor,
    gate_logits: Tensor,
    expert_fn: Callable[[Tensor], Tensor],
    min_experts: int,
    max_experts: int,
    high_conf_threshold: float,
) -> Tensor:
    """Route ``x`` with an adaptive per-token expert count and combine.

    Every token is scored against the top ``max_experts`` experts once; a
    confident token retains only its ``min_experts`` (renormalized), an
    uncertain token its ``max_experts``. ``expert_fn`` maps the broadcast
    ``[T, max_experts, d]`` selected slots to ``[T, max_experts, d]`` (the
    per-expert block applied to the token). Returns the combined ``[T, d]``.
    """
    num_tokens, num_experts = gate_logits.shape
    max_experts = min(max_experts, num_experts)
    min_experts = min(min_experts, max_experts)

    prob = torch.softmax(gate_logits, dim=-1)  # [T, E]
    k_t = dynamic_expert_count(gate_logits, min_experts, max_experts, high_conf_threshold)  # [T]
    # Rank the top max_experts once; keep the first k_t per token.
    scores, _ = prob.topk(max_experts, dim=-1)  # [T, max]
    keep = torch.arange(max_experts, device=gate_logits.device)[None, :] < k_t.unsqueeze(-1)  # [T, max]
    masked = scores.masked_fill(~keep, float("-inf"))
    weights = torch.softmax(masked, dim=-1)  # renormalize over kept experts

    slots = x[:, None, :].expand(num_tokens, max_experts, x.shape[-1])  # [T, max, d]
    expert_out = expert_fn(slots)  # [T, max, d]
    return (weights.unsqueeze(-1) * expert_out).sum(dim=1)  # [T, d]
