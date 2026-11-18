"""Sampling-parameter validation.

Covers two deep-dive findings on :mod:`llm.generation.sampling`:

: ``sampling_probs`` silently ignored an out-of-range ``top_p``
  (``<= 0`` or ``>= 1``) instead of rejecting it like ``temperature`` and
  ``top_k`` — ``top_p=0`` (the greedy analog) drew non-deterministically.
: ``apply_repetition_penalty`` bypassed the ``<= 0`` ValueError
  when ``token_ids`` was empty, because the guard sat after the empty
  early-return — a direct-API inconsistency (the generator paths always
  pass a non-empty history).
: the served OpenAI surface accepts ``top_p=1.0`` ("1.0 means no
  truncation", the OpenAI client default) but the ``>= 1.0`` rejection made
  every such request error. ``top_p >= 1.0`` now means "no nucleus
  filtering" (a no-op, like ``None``); only ``top_p <= 0`` is rejected.
"""

import pytest
import torch

from llm.generation.sampling import apply_repetition_penalty, sampling_probs

# --: top_p range validation in sampling_probs -----------


@pytest.mark.parametrize("bad_top_p", [0.0, -0.5])
def test_sampling_probs_rejects_nonpositive_top_p(bad_top_p):
    """``sampling_probs`` must reject ``top_p <= 0`` like it rejects bad
    ``temperature`` / ``top_k`` instead of silently skipping
    the nucleus filter and drawing from the full distribution — ``top_p=0``
    (the greedy analog) would draw non-deterministically."""
    with pytest.raises(ValueError, match="top_p"):
        sampling_probs(torch.zeros(8), top_p=bad_top_p)


def test_sampling_probs_accepts_valid_top_p_range_edges():
    """Valid in-range ``top_p`` still works (no false rejection)."""
    probs = sampling_probs(torch.zeros(8), top_p=0.98)
    assert probs.shape == (8,)
    assert torch.isclose(probs.sum(), torch.ones(()), atol=1e-6)


@pytest.mark.parametrize("top_p", [1.0, 1.5])
def test_sampling_probs_top_p_one_or_above_is_noop(top_p):
    """``top_p >= 1.0`` must NOT raise: it is the OpenAI client default and
    the serving schema's documented "1.0 means no truncation".
    It behaves exactly like ``top_p=None`` — the filter is skipped, so the
    returned distribution is the full softmax."""
    base = sampling_probs(torch.zeros(8), top_p=None)
    assert torch.equal(sampling_probs(torch.zeros(8), top_p=top_p), base)


# --: repetition penalty guard order -------------------------------


def test_repetition_penalty_empty_token_ids_still_validates():
    """A non-positive ``repetition_penalty`` must raise even when
    ``token_ids`` is empty — the ``<= 0`` guard used to sit
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


# --- cross-validation against transformers' own warpers -------------------
#
# The tests above pin validation errors and structural invariants. None of them
# checks the DISTRIBUTION against an independent implementation, so a change to
# the filter order, the nucleus right-shift, or the per-row threshold would keep
# them green while every backend sampled from the wrong distribution.
#
# `transformers` ships the same three operations as LogitsWarper classes; they
# are the reference this module's semantics were chosen to match, and the
# project already depends on the package for the HF compatibility layer.


def _hf_distribution(logits, *, temperature, top_k, top_p):
    """Reference distribution from transformers' warpers, applied in the same order."""
    transformers = pytest.importorskip("transformers", reason="transformers is the reference implementation")
    torch = pytest.importorskip("torch")

    warped = logits.clone().float().unsqueeze(0)
    sentinel_ids = torch.zeros(1, 1, dtype=torch.long)
    if temperature != 1.0:
        warped = transformers.TemperatureLogitsWarper(temperature)(sentinel_ids, warped)
    if top_k is not None:
        warped = transformers.TopKLogitsWarper(top_k)(sentinel_ids, warped)
    if top_p is not None:
        warped = transformers.TopPLogitsWarper(top_p)(sentinel_ids, warped)
    return torch.softmax(warped.squeeze(0), dim=-1)


@pytest.mark.parametrize(
    ("top_k", "top_p", "temperature"),
    [
        (3, None, 1.0),
        (None, 0.9, 1.0),
        (5, 0.9, 1.0),
        (4, 0.8, 0.7),
        (None, 0.5, 2.0),
    ],
)
def test_sampling_probs_matches_transformers_warpers(top_k, top_p, temperature):
    """The filtered distribution must equal transformers' warper pipeline exactly."""
    torch = pytest.importorskip("torch")
    from llm.generation.sampling import sampling_probs

    torch.manual_seed(0)
    logits = torch.randn(16)

    got = sampling_probs(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    expected = _hf_distribution(logits, temperature=temperature, top_k=top_k, top_p=top_p)

    assert torch.allclose(got, expected, atol=1e-6), (
        f"top_k={top_k} top_p={top_p} temperature={temperature}: "
        f"max deviation {float((got - expected).abs().max()):.3e}"
    )


def test_the_batched_path_uses_each_rows_own_threshold():
    """A ``[gamma, vocab]`` input must filter each row independently.

    This is the shape the speculative backend scores against. Sharing one row's
    top-k threshold across the batch (the historical bug) produces a valid
    distribution that sums to 1 per row, so only comparing against the
    per-row reference catches it.
    """
    torch = pytest.importorskip("torch")
    from llm.generation.sampling import sampling_probs

    torch.manual_seed(0)
    gamma, vocab = 6, 20
    logits = torch.randn(gamma, vocab)

    got = sampling_probs(logits, top_k=5, top_p=0.85, temperature=1.1)
    expected = torch.stack(
        [_hf_distribution(logits[row], temperature=1.1, top_k=5, top_p=0.85) for row in range(gamma)]
    )

    assert torch.allclose(got, expected, atol=1e-6), (
        f"batched rows do not match their own references; max deviation {float((got - expected).abs().max()):.3e}"
    )
    # Independent sanity: masking must leave each row a proper distribution.
    assert torch.allclose(got.sum(dim=-1), torch.ones(gamma), atol=1e-6)


def test_a_top_k_wider_than_the_vocabulary_still_normalises():
    """``top_k > vocab_size`` must clamp, not raise or produce an unnormalised row."""
    torch = pytest.importorskip("torch")
    from llm.generation.sampling import sampling_probs

    torch.manual_seed(0)
    logits = torch.randn(10)

    got = sampling_probs(logits, top_k=999)

    assert abs(float(got.sum()) - 1.0) < 1e-6
    # With every token surviving, the result is the plain softmax.
    assert torch.allclose(got, torch.softmax(logits.float(), dim=-1), atol=1e-6)


def test_half_precision_logits_do_not_overflow_at_a_small_temperature():
    """fp16 logits with a tiny temperature must widen rather than produce NaN.

    In fp16 the division overflows around 65504, so the softmax returned NaN and
    ``torch.multinomial`` raised an error naming neither the temperature nor the
    dtype.
    """
    torch = pytest.importorskip("torch")
    from llm.generation.sampling import sampling_probs

    torch.manual_seed(0)
    logits = (torch.randn(10) * 1.5).half()

    got = sampling_probs(logits, temperature=1e-5)

    assert torch.isfinite(got).all(), "fp16 logits with a small temperature produced non-finite probabilities"
    assert abs(float(got.sum()) - 1.0) < 1e-5
    # A tiny temperature concentrates on the argmax, which must be preserved.
    assert int(got.argmax()) == int(logits.float().argmax())
