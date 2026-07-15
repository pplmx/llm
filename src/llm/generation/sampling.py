"""Shared token sampling utilities for generation backends."""

from __future__ import annotations

from collections import Counter

import torch


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty in-place on 1D logits."""
    if repetition_penalty == 1.0 or not token_ids:
        return logits

    adjusted = logits.clone()
    device = adjusted.device
    ids = torch.tensor(token_ids, device=device)
    scores = torch.gather(adjusted, 0, ids)
    scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
    adjusted.scatter_(0, ids, scores)
    return adjusted


def apply_frequency_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    frequency_penalty: float,
) -> torch.Tensor:
    """Subtract ``frequency_penalty * count(token)`` from each seen token's logit.

    Implements the OpenAI-compatible ``frequency_penalty`` semantics
    (see https://platform.openai.com/docs/api-reference/chat/create):
    positive values penalise tokens in proportion to how often they
    have already appeared in the generated text. Zero (the default)
    is a no-op so callers don't need to special-case the off state.

    Args:
        logits: 1D ``[vocab_size]`` tensor. Not mutated.
        token_ids: List of token ids generated so far (may include
            duplicates; duplicates count toward the penalty).
        frequency_penalty: Penalty coefficient. ``0.0`` disables the
            adjustment; values typically live in ``[-2.0, 2.0]``.

    Returns:
        A new 1D tensor with the per-frequency penalty subtracted.
    """
    if frequency_penalty == 0.0 or not token_ids:
        return logits

    counts = Counter(token_ids)
    vocab_size = logits.size(-1)
    # Drop ids that fall outside the vocab — they're not representable
    # in these logits, so penalising them is meaningless and would
    # raise an index error on the scatter below.
    valid_ids = {tid: c for tid, c in counts.items() if 0 <= tid < vocab_size}
    if not valid_ids:
        return logits

    adjusted = logits.clone()
    device = adjusted.device
    ids = torch.tensor(list(valid_ids), device=device, dtype=torch.long)
    penalties = torch.tensor([valid_ids[tid] for tid in valid_ids], device=device, dtype=adjusted.dtype)
    adjusted.scatter_add_(
        0,
        ids,
        -frequency_penalty * penalties,
    )
    return adjusted


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Sample one token id from 1D logits."""
    if temperature == 0:
        return int(torch.argmax(logits, dim=-1).item())

    next_logits = logits / temperature

    if top_k is not None:
        vocab_size = next_logits.size(-1)
        values, _ = torch.topk(next_logits, min(top_k, vocab_size))
        next_logits = next_logits.clone()
        next_logits[next_logits < values[-1]] = -torch.inf

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        next_logits = next_logits.clone()
        next_logits[sorted_indices[sorted_indices_to_remove]] = -float("inf")

    probs = torch.softmax(next_logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
