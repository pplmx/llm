"""Tests for shared generation sampling helpers."""

import pytest
import torch

from llm.generation.sampling import (
    apply_frequency_penalty,
    apply_logit_bias,
    apply_penalties,
    apply_presence_penalty,
    apply_repetition_penalty,
    sample_next_token,
    sampling_probs,
)


def test_greedy_sampling():
    logits = torch.tensor([0.1, 2.0, 0.5])
    assert sample_next_token(logits, temperature=0.0) == 1


def test_negative_temperature_rejected():
    """a negative temperature silently inverts the logits and
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


def test_invalid_penalty_and_topk_rejected():
    """repetition_penalty <= 0 silently corrupts logits (0 divides
    by zero -> inf, negative flips sign) and top_k <= 0 crashes with a cryptic
    IndexError; both are now rejected with clear ValueErrors. Legitimate values
    ((0, 1] repetition, top_k >= 1) still work."""
    logits = torch.tensor([1.0, 2.0, -1.0])
    with pytest.raises(ValueError, match="repetition_penalty"):
        apply_repetition_penalty(logits, [0, 1, 2], repetition_penalty=0.0)
    with pytest.raises(ValueError, match="repetition_penalty"):
        apply_repetition_penalty(logits, [0, 1, 2], repetition_penalty=-1.0)
    with pytest.raises(ValueError, match="top_k"):
        sampling_probs(torch.tensor([1.0, 2.0, 3.0]), temperature=1.0, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        sampling_probs(torch.tensor([1.0, 2.0, 3.0]), temperature=1.0, top_k=-2)
    # Valid values still produce finite, non-empty distributions.
    assert torch.isfinite(apply_repetition_penalty(logits, [0, 1, 2], repetition_penalty=2.0)).all()
    assert sampling_probs(torch.tensor([1.0, 2.0, 3.0]), temperature=1.0, top_k=2).shape[0] == 3


def test_sampling_probs_batched_topk_topp_per_row():
    """``sampling_probs`` must apply top-k/top-p filtering PER ROW on batched
    ``[gamma, vocab]`` logits (the speculative backend passes one row per draft
    position).

    Regression: the old masking was written for 1D only — ``values[-1]``
    picked one row's top-k threshold (broadcast crash for ``k != vocab``) and
    ``sorted_indices_to_remove[1:]`` shifted along the ROW axis (IndexError
    for ``top_p``), silently disabling the filter for a single row. The
    speculative acceptance score is then computed against the unfiltered
    distribution — the exact divergence ``test_speculative_top_k_acceptance_uses_filtered_distributions``
    guards against.
    """
    gamma, vocab = 3, 8
    # Tie-free rows (distinct peak / second per row) so the top-2 threshold is
    # unambiguous — with tied logits equal to the threshold, top-k keeps every
    # tie (correct semantics), which would make this test vacuous.
    logits = torch.full((gamma, vocab), 0.0)
    for row in range(gamma):
        logits[row, row + 1] = 5.0  # rows peak at different vocab ids
        logits[row, 6] = 3.0  # unambiguous second-largest

    probs_topk = sampling_probs(logits, temperature=1.0, top_k=2)
    # Per-row nonzero count == top_k for every row.
    assert (probs_topk > 0).sum(dim=-1).tolist() == [2, 2, 2]
    # mask must be row-local: row r's survivors include its own peak.
    for row in range(gamma):
        assert probs_topk[row, row + 1] > 0

    probs_topp = sampling_probs(logits + 0.1, temperature=1.0, top_p=0.5)
    assert torch.allclose(probs_topp.sum(dim=-1), torch.ones(gamma), atol=1e-6)
    # top_p=0.5 must actually shrink each row (not silently return the full softmax).
    full = torch.softmax(logits + 0.1, dim=-1)
    assert not torch.allclose(probs_topp, full)
    # And a 1D input still produces the identical distribution as a 1-row batch.
    one_row = logits[0] + 0.1
    assert torch.allclose(
        sampling_probs(one_row, temperature=1.0, top_p=0.5),
        sampling_probs(one_row.unsqueeze(0), temperature=1.0, top_p=0.5).squeeze(0),
    )


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


def test_apply_penalties_fused_equals_chained():
    """The fused :func:`apply_penalties` (used by the decode hot paths) must be
    bit-identical to applying the four helpers in sequence — a later fusion
    rewrite must not drift the serving/generation output."""
    rng = torch.Generator().manual_seed(7)
    cases = [
        # (vocab, penalty args)
        (16, {"repetition_penalty": 1.5}),
        (16, {"frequency_penalty": 0.5}),
        (16, {"presence_penalty": 0.5}),
        (16, {"repetition_penalty": 0.5, "frequency_penalty": -1.0, "presence_penalty": 0.3}),
        (16, {"logit_bias": {0: 2.5, 5: -1.0}}),
        (
            16,
            {
                "repetition_penalty": 2.0,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.2,
                "logit_bias": {2: 3.0, "4": -0.5},
            },
        ),
    ]
    for vocab, kwargs in cases:
        logits = torch.randn(vocab, generator=rng) * 3
        # history with duplicates, an out-of-vocab id, and (one case) empty
        token_ids = [3, 1, 3, 0, 7, 99] if kwargs.get("repetition_penalty") else [3, 1, 3, 5, 5]
        chained = logits
        if kwargs.get("repetition_penalty", 1.0) != 1.0:
            chained = apply_repetition_penalty(chained, token_ids, repetition_penalty=kwargs["repetition_penalty"])
        if kwargs.get("frequency_penalty", 0.0) != 0.0:
            chained = apply_frequency_penalty(chained, token_ids, frequency_penalty=kwargs["frequency_penalty"])
        if kwargs.get("presence_penalty", 0.0) != 0.0:
            chained = apply_presence_penalty(chained, token_ids, presence_penalty=kwargs["presence_penalty"])
        if kwargs.get("logit_bias"):
            chained = apply_logit_bias(chained, kwargs["logit_bias"])
        fused = apply_penalties(logits, token_ids, **kwargs)
        assert torch.equal(fused, chained)

    # No active penalty -> returns the identical object (mirrors chaining no-ops).
    x = torch.randn(8)
    assert apply_penalties(x, [1, 2]) is x
    # Empty history with a bias still applies the bias.
    biased = apply_penalties(torch.zeros(4), [], logit_bias={1: 2.0})
    assert biased[1].item() == 2.0
    # repetition_penalty <= 0 must raise even with an empty history.
    with pytest.raises(ValueError, match="repetition_penalty"):
        apply_penalties(torch.zeros(8), [], repetition_penalty=0.0)
    with pytest.raises(ValueError, match="repetition_penalty"):
        apply_penalties(torch.zeros(8), [1], repetition_penalty=-1.0)


class TestCombinedTopKAndTopP:
    """Both filters can be set at once; the result must be their INTERSECTION.

    `top_k` keeps the k largest logits and `top_p` keeps the smallest set whose
    cumulative probability exceeds the threshold. Applying them in either order
    leaves exactly the surviving tokens that pass BOTH — so the effective
    support is the smaller of the two. Setting them together is legal in the
    OpenAI API and nothing was checking the interaction, which is where an
    implementation can silently drop one of the two filters.
    """

    def test_effective_support_is_the_intersection(self):
        import torch

        from llm.generation.sampling import sampling_probs

        torch.manual_seed(0)
        logits = torch.randn(20) * 2

        k_only = sampling_probs(logits, temperature=1.0, top_k=5)
        p_only = sampling_probs(logits, temperature=1.0, top_p=0.9)
        both = sampling_probs(logits, temperature=1.0, top_k=5, top_p=0.9)

        k_support = set(torch.nonzero(k_only > 0).flatten().tolist())
        p_support = set(torch.nonzero(p_only > 0).flatten().tolist())
        both_support = set(torch.nonzero(both > 0).flatten().tolist())

        assert both_support == k_support & p_support, (
            f"combined support {sorted(both_support)} is not the intersection "
            f"of top_k {sorted(k_support)} and top_p {sorted(p_support)}"
        )
        assert both.sum().item() == pytest.approx(1.0, abs=1e-5), "the surviving mass must still be normalised"

    def test_combined_filter_never_widens_the_support(self):
        """Setting BOTH must never keep MORE tokens than either filter alone.

        A bug that applied only the looser of the two, or reset the mask between
        them, would widen the support relative to the stricter single filter.
        """
        import torch

        from llm.generation.sampling import sampling_probs

        torch.manual_seed(0)
        logits = torch.randn(32) * 2

        for top_k, top_p in ((5, 0.9), (10, 0.5), (3, 0.99)):
            combined = sampling_probs(logits, temperature=1.0, top_k=top_k, top_p=top_p)
            k_only = sampling_probs(logits, temperature=1.0, top_k=top_k)
            p_only = sampling_probs(logits, temperature=1.0, top_p=top_p)

            combined_count = int((combined > 0).sum())
            assert combined_count <= int((k_only > 0).sum()), f"top_p widened the top_k support for {top_k}/{top_p}"
            assert combined_count <= int((p_only > 0).sum()), f"top_k widened the top_p support for {top_k}/{top_p}"


@pytest.mark.parametrize("bad_top_p", [0.0, -0.5])
def test_sampling_probs_rejects_nonpositive_top_p(bad_top_p):
    """``top_p <= 0`` must be rejected, not silently ignored.

    ``sampling_probs`` already rejects a non-positive ``temperature`` and
    ``top_k < 1`` here. ``top_p`` was the sibling case that had its guard but
    no assertion **in this file** — the only test for it lived in
    ``tests/generation/test_sampling_validation.py``. Verified by mutation:
    weakening the guard to ``top_p < 0`` is caught when both directories run,
    but SURVIVES when this file runs alone, so the protection depended on which
    test subset a change happened to be checked against.

    ``top_p=0`` is the degenerate "greedy analog" that would otherwise draw
    non-deterministically: the nucleus filter's ``0.0 < top_p < 1.0`` condition
    skips it and the draw falls back to the full distribution.
    """
    with pytest.raises(ValueError, match="top_p"):
        sampling_probs(torch.zeros(8), top_p=bad_top_p)


@pytest.mark.parametrize("top_p", [1.0, 1.5, None])
def test_sampling_probs_accepts_top_p_at_or_above_one(top_p):
    """ "1.0 means no truncation" (the OpenAI client default) must not raise.

    ``top_p >= 1.0`` is explicitly valid and behaves like ``None``: the full
    distribution survives. Rejecting it made every default-configured OpenAI
    request fail, which is why the boundary is pinned in both directions — a
    guard that rejects everything non-``None`` would pass the rejection test
    above.
    """
    probs = sampling_probs(torch.zeros(8), top_p=top_p)
    assert probs.shape == (8,)
    assert torch.isclose(probs.sum(), torch.ones(()), atol=1e-6)
    assert (probs > 0).all(), "top_p >= 1.0 must not mask any token"
