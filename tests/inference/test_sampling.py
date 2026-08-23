"""Tests for shared generation sampling helpers."""

import pytest
import torch

from llm.generation.sampling import apply_repetition_penalty, sample_next_token, sampling_probs


def test_greedy_sampling():
    logits = torch.tensor([0.1, 2.0, 0.5])
    assert sample_next_token(logits, temperature=0.0) == 1


def test_negative_temperature_rejected():
    """RIL TASK-249: a negative temperature silently inverts the logits and
    turns the sampler anti-greedy (draws the lowest-logit token); it must be
    rejected instead of silently corrupting generation. Sampling at
    temperature 0 (greedy) remains valid; a positive temperature still works."""
    logits = torch.tensor([10.0, 0.0, 5.0])
    with pytest.raises(ValueError, match="temperature"):
        sample_next_token(logits, temperature=-1.0)
    with pytest.raises(ValueError, match="temperature"):
        sampling_probs(logits, temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        sampling_probs(logits, temperature=-2.0)
    # Greedy (0) stays supported; a normal sampler still yields the argmax
    # token as the most probable draw.
    assert sample_next_token(logits, temperature=0.0) == 0
    assert int(sampling_probs(logits, temperature=1.0).argmax()) == 0


def test_repetition_penalty_changes_logits():
    logits = torch.tensor([1.0, 2.0, 3.0])
    adjusted = apply_repetition_penalty(logits, [1, 2], repetition_penalty=2.0)
    assert adjusted[1].item() != logits[1].item()


def test_repetition_penalty_ignores_tokens_outside_vocab():
    """Token ids outside [0, vocab_size) must not crash gather/scatter.

    Regression test: with an id not representable in the logits (e.g. a
    truncation or an API boundary passing an id the model's vocabulary
    never produced), the previous implementation raised an
    out-of-bounds error from ``torch.gather``. Such ids are now silently
    dropped, matching frequency/presence penalty helpers.
    """
    logits = torch.tensor([1.0, 2.0, 3.0])
    snapshot = logits.clone()
    # All ids out of range -> no penalty applies, logits unchanged.
    out = apply_repetition_penalty(logits, [5, 7, 100], repetition_penalty=2.0)
    assert torch.equal(out, logits)
    # Single out-of-range id only -> unchanged, no crash.
    out = apply_repetition_penalty(logits, [99], repetition_penalty=2.0)
    assert torch.equal(out, logits)

    # Mixed valid + invalid: invalid ids are dropped, only the valid id
    # [1] is penalised (2.0 -> /2.0 = 1.0).
    adjusted = apply_repetition_penalty(logits, [1, 99], repetition_penalty=2.0)
    assert torch.allclose(adjusted[1], torch.tensor(1.0))
    assert torch.allclose(adjusted[0], torch.tensor(1.0))
    assert torch.allclose(adjusted[2], torch.tensor(3.0))
    # Input is not mutated.
    assert torch.equal(logits, snapshot)
