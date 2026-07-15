"""Tests for the frequency_penalty sampling helper (Tier 3 #35)."""

from __future__ import annotations

import torch

from llm.generation.sampling import apply_frequency_penalty


def test_apply_frequency_penalty_zero_is_noop():
    """Zero penalty leaves logits unchanged (returns same values)."""
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = apply_frequency_penalty(logits, [0, 1], frequency_penalty=0.0)
    assert torch.equal(out, logits)


def test_apply_frequency_penalty_empty_token_ids_is_noop():
    """No generated tokens → no penalty (no count to multiply by)."""
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = apply_frequency_penalty(logits, [], frequency_penalty=2.0)
    assert torch.equal(out, logits)


def test_apply_frequency_penalty_reduces_logits_in_proportion_to_count():
    """Token with count=2 is penalised twice as much as count=1 (other things equal)."""
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # token 0 appears once, token 1 appears twice; tokens 2 and 3 absent
    out = apply_frequency_penalty(logits.clone(), [0, 1, 1], frequency_penalty=0.5)

    # expected: logits[i] - 0.5 * count[i] (0 for absent tokens)
    assert torch.allclose(out[0], torch.tensor(0.5))  # 1.0 - 0.5*1
    assert torch.allclose(out[1], torch.tensor(1.0))  # 2.0 - 0.5*2
    assert torch.allclose(out[2], torch.tensor(3.0))  # 3.0 - 0.0
    assert torch.allclose(out[3], torch.tensor(4.0))  # 4.0 - 0.0


def test_apply_frequency_penalty_does_not_mutate_input():
    """The helper returns a new tensor — input logits are unchanged."""
    logits = torch.tensor([1.0, 2.0, 3.0])
    snapshot = logits.clone()
    _ = apply_frequency_penalty(logits, [0, 0], frequency_penalty=1.0)
    assert torch.equal(logits, snapshot)


def test_apply_frequency_penalty_ignores_tokens_outside_vocab():
    """Token ids >= vocab_size are silently ignored (matches apply_repetition_penalty)."""
    logits = torch.tensor([1.0, 2.0])
    out = apply_frequency_penalty(logits, [5, 7], frequency_penalty=10.0)
    # No id falls inside [0, vocab_size), so no penalty applies.
    assert torch.equal(out, logits)
