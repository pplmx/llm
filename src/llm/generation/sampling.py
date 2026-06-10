"""Shared token sampling utilities for generation backends."""

from __future__ import annotations

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
