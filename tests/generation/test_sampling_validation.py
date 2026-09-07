"""Sampling-parameter validation (RIL ISS-390 / ISS-391).

Covers two round-180 deep-dive findings on :mod:`llm.generation.sampling`:

- ISS-390: ``sampling_probs`` silently ignored an out-of-range ``top_p``
  (``<= 0`` or ``>= 1``) instead of rejecting it like ``temperature`` and
  ``top_k`` — ``top_p=0`` (the greedy analog) drew non-deterministically.
- ISS-391: ``apply_repetition_penalty`` bypassed the ``<= 0`` ValueError
  when ``token_ids`` was empty, because the guard sat after the empty
  early-return — a direct-API inconsistency (the generator paths always
  pass a non-empty history).
"""

import pytest
import torch

from llm.generation.sampling import apply_repetition_penalty, sampling_probs

# --- ISS-390: top_p range validation in sampling_probs --------------------


@pytest.mark.parametrize("bad_top_p", [0.0, -0.5, 1.0, 1.5])
def test_sampling_probs_rejects_out_of_range_top_p(bad_top_p):
    """``sampling_probs`` must reject ``top_p`` outside (0, 1) like it rejects
    bad ``temperature`` / ``top_k`` (RIL ISS-390) instead of silently
    skipping the nucleus filter and drawing from the full distribution."""
    with pytest.raises(ValueError, match="top_p"):
        sampling_probs(torch.zeros(8), top_p=bad_top_p)


def test_sampling_probs_accepts_valid_top_p_range_edges():
    """Valid in-range ``top_p`` still works (no false rejection)."""
    probs = sampling_probs(torch.zeros(8), top_p=0.98)
    assert probs.shape == (8,)
    assert torch.isclose(probs.sum(), torch.ones(()), atol=1e-6)


# --- ISS-391: repetition penalty guard order -------------------------------


def test_repetition_penalty_empty_token_ids_still_validates():
    """A non-positive ``repetition_penalty`` must raise even when
    ``token_ids`` is empty (RIL ISS-391) — the ``<= 0`` guard used to sit
    after the ``not token_ids`` early-return, so ``penalty=0`` / negative
    silently passed on the empty-history path."""
    with pytest.raises(ValueError, match="repetition_penalty"):
        apply_repetition_penalty(torch.zeros(8), token_ids=[], repetition_penalty=0)


@pytest.mark.parametrize("penalty", [0.0, -1.0])
def test_repetition_penalty_nonpositive_rejected_with_history(penalty):
    """Sanity: with a non-empty history the guard already fired (pins the
    existing contract so the empty-path fix can't drift)."""
    with pytest.raises(ValueError, match="repetition_penalty"):
        apply_repetition_penalty(torch.zeros(8), token_ids=[1, 2], repetition_penalty=penalty)
