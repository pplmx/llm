"""Property-based tests for :mod:`llm.generation.sampling` (Finding AD).

The sampling module is the heart of every generation backend. Two
correctness invariants must hold for arbitrary inputs:

1. :func:`sample_next_token` returns a value in ``[0, vocab_size)``.
2. :func:`sample_next_token` with ``temperature=0`` is deterministic —
   it returns ``argmax`` regardless of vocabulary size or logit
   distribution.
3. :func:`sample_next_token` with ``top_k=k`` never returns an index
   outside the top-k entries of the input logits.
4. :func:`apply_repetition_penalty` with ``penalty=1.0`` is the
   identity function (no token in the list was penalised).

Hypothesis tries tens of vocab sizes, logit distributions, and token
lists per test — corner cases the example-based tests do not reach.
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from llm.generation.sampling import apply_repetition_penalty, sample_next_token

# Strategy: 1D logits of arbitrary vocab size (small enough to keep
# multinomial cheap, large enough to exercise top-k / top-p).
_VOCAB = st.integers(min_value=4, max_value=64)
_LOGITS = _VOCAB.flatmap(
    lambda v: st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=v,
        max_size=v,
    ).map(torch.tensor)
)


# --- Invariant 1: sample_next_token returns a valid index -----------------


@given(logits=_LOGITS, temperature=st.floats(min_value=0.01, max_value=2.0, allow_nan=False))
@settings(max_examples=50, deadline=None)
def test_sample_returns_index_in_range(logits, temperature):
    """``sample_next_token`` returns ``int`` in ``[0, vocab_size)``.

    Out-of-range indices would crash downstream embedding lookups;
    Hypothesis exercises many vocab sizes to catch off-by-one in the
    multinomial draw.
    """
    vocab_size = logits.size(-1)
    sampled = sample_next_token(logits, temperature=temperature)
    assert isinstance(sampled, int)
    assert 0 <= sampled < vocab_size


# --- Invariant 2: temperature=0 returns argmax ----------------------------


@given(logits=_LOGITS)
@settings(max_examples=50, deadline=None)
def test_temperature_zero_is_argmax(logits):
    """``temperature=0`` is the greedy path: return ``argmax``.

    Greedy decoding must be fully deterministic — a flake here means
    the serving tier occasionally picks a non-greedy token under
    temperature=0, breaking eval pipelines that compare against the
    reference implementation.
    """
    sampled = sample_next_token(logits, temperature=0.0)
    expected = int(torch.argmax(logits, dim=-1).item())
    assert sampled == expected


# --- Invariant 3: top_k restricts the sample space ------------------------


@given(
    logits=_LOGITS,
    top_k=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=50, deadline=None)
def test_top_k_restricts_to_top_k_indices(logits, top_k):
    """With ``top_k=k``, the sampled index is one of the top-k logits.

    Top-k must truly restrict the distribution: a sample outside the
    top-k would mean the mask is being computed from a stale logits
    tensor (post-softmax) or applied to the wrong axis.
    """
    vocab_size = logits.size(-1)
    effective_k = min(top_k, vocab_size)
    # Find the threshold value (smallest of the top-k logits).
    _, top_indices = torch.topk(logits, effective_k)
    threshold = logits[top_indices[-1]].item()

    # Sample many times and assert every draw is in the top-k.
    for _ in range(20):
        sampled = sample_next_token(logits, temperature=1.0, top_k=top_k)
        assert logits[sampled].item() >= threshold - 1e-6, (
            f"sampled index {sampled} has logit {logits[sampled].item()} "
            f"below top_k threshold {threshold}"
        )


# --- Invariant 4: penalty=1.0 is identity ---------------------------------


@given(
    logits=_LOGITS,
    token_ids=st.lists(st.integers(min_value=0, max_value=63), min_size=0, max_size=10),
)
@settings(max_examples=50, deadline=None)
def test_repetition_penalty_one_is_identity(logits, token_ids):
    """``apply_repetition_penalty(..., penalty=1.0)`` returns unchanged logits.

    A no-op penalty is the common case at inference time (no
    repetition penalty requested). It must produce bit-exact output.
    """
    # Filter ids to the actual vocab size so the gather doesn't fail.
    vocab_size = logits.size(-1)
    safe_ids = [t for t in token_ids if 0 <= t < vocab_size]
    out = apply_repetition_penalty(logits.clone(), safe_ids, repetition_penalty=1.0)
    assert torch.equal(out, logits)


# --- Invariant 5: repetition penalty strictly down-weights repeated tokens


@given(
    vocab=st.integers(min_value=4, max_value=16),
    target_logit=st.floats(min_value=0.5, max_value=5.0, allow_nan=False),
    penalty=st.floats(min_value=1.01, max_value=4.0, allow_nan=False),
)
@settings(max_examples=50, deadline=None)
def test_repetition_penalty_divides_positive_logit(vocab, target_logit, penalty):
    """A positive logit at a repeated id is divided by ``penalty``.

    Penalty > 1 must monotonically decrease the chosen id's logit
    (for positive logits). Otherwise the model would actually
    reinforce repeated tokens, exactly the opposite of the contract.
    """
    logits = torch.zeros(vocab)
    target_id = 0
    logits[target_id] = target_logit

    out = apply_repetition_penalty(logits.clone(), [target_id], repetition_penalty=penalty)
    # Positive logit should be divided by penalty.
    expected = target_logit / penalty
    assert abs(out[target_id].item() - expected) < 1e-5
    assert out[target_id].item() < target_logit


# --- Invariant 6: repetition penalty amplifies negative logits -----------


@given(
    vocab=st.integers(min_value=4, max_value=16),
    target_logit=st.floats(min_value=-5.0, max_value=-0.5, allow_nan=False),
    penalty=st.floats(min_value=1.01, max_value=4.0, allow_nan=False),
)
@settings(max_examples=50, deadline=None)
def test_repetition_penalty_multiplies_negative_logit(vocab, target_logit, penalty):
    """A negative logit at a repeated id is multiplied by ``penalty``.

    Mirrors invariant 5: penalties must push negative logits further
    from zero (more negative). If the multiplication is missing, the
    model would still occasionally sample already-used tokens.
    """
    logits = torch.zeros(vocab)
    target_id = 0
    logits[target_id] = target_logit

    out = apply_repetition_penalty(logits.clone(), [target_id], repetition_penalty=penalty)
    expected = target_logit * penalty
    assert abs(out[target_id].item() - expected) < 1e-5
    assert out[target_id].item() < target_logit  # more negative than input
