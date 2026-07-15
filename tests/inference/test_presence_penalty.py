"""Tests for the presence_penalty sampling helper (Tier 3 #37)."""

from __future__ import annotations

import torch

from llm.generation.sampling import apply_presence_penalty


def test_apply_presence_penalty_zero_is_noop():
    """Zero penalty leaves logits unchanged."""
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = apply_presence_penalty(logits, [0, 1], presence_penalty=0.0)
    assert torch.equal(out, logits)


def test_apply_presence_penalty_empty_token_ids_is_noop():
    """No generated tokens → no penalty."""
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = apply_presence_penalty(logits, [], presence_penalty=2.0)
    assert torch.equal(out, logits)


def test_apply_presence_penalty_subtracts_flat_regardless_of_count():
    """A token that appears 5 times is penalised the same as one that appears once.

    This is the key distinction from ``apply_frequency_penalty``:
    presence_penalty cares only about presence, not frequency.
    """
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # token 0 appears 5 times, token 1 appears once; tokens 2 and 3 absent
    out = apply_presence_penalty(logits.clone(), [0, 0, 0, 0, 0, 1], presence_penalty=0.7)

    # Both seen tokens drop by exactly 0.7 — count doesn't matter.
    assert torch.allclose(out[0], torch.tensor(0.3))  # 1.0 - 0.7
    assert torch.allclose(out[1], torch.tensor(1.3))  # 2.0 - 0.7
    # Unseen tokens are untouched.
    assert torch.allclose(out[2], torch.tensor(3.0))
    assert torch.allclose(out[3], torch.tensor(4.0))


def test_apply_presence_penalty_can_boost_when_negative():
    """Negative presence_penalty adds (boosts) seen tokens — OpenAI semantics."""
    logits = torch.tensor([1.0, 2.0, 3.0])
    out = apply_presence_penalty(logits.clone(), [0], presence_penalty=-0.5)
    assert torch.allclose(out[0], torch.tensor(1.5))  # 1.0 - (-0.5) = 1.5
    assert torch.allclose(out[1], torch.tensor(2.0))  # untouched
    assert torch.allclose(out[2], torch.tensor(3.0))  # untouched


def test_apply_presence_penalty_does_not_mutate_input():
    """The helper returns a new tensor — input logits are unchanged."""
    logits = torch.tensor([1.0, 2.0, 3.0])
    snapshot = logits.clone()
    _ = apply_presence_penalty(logits, [0, 0], presence_penalty=1.0)
    assert torch.equal(logits, snapshot)


def test_apply_presence_penalty_ignores_tokens_outside_vocab():
    """Token ids >= vocab_size are silently ignored (mirrors apply_frequency_penalty)."""
    logits = torch.tensor([1.0, 2.0])
    out = apply_presence_penalty(logits, [5, 7], presence_penalty=10.0)
    # No id falls inside [0, vocab_size), so no penalty applies.
    assert torch.equal(out, logits)
