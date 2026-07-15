"""Tests for the logit_bias sampling helper (Tier 3 #38)."""

from __future__ import annotations

import torch

from llm.generation.sampling import apply_logit_bias


def test_apply_logit_bias_none_is_noop():
    """None logit_bias leaves logits unchanged."""
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = apply_logit_bias(logits, None)
    assert torch.equal(out, logits)


def test_apply_logit_bias_empty_dict_is_noop():
    """Empty logit_bias leaves logits unchanged."""
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = apply_logit_bias(logits, {})
    assert torch.equal(out, logits)


def test_apply_logit_bias_adds_per_token_amount():
    """Each affected token's logit is shifted by exactly its bias."""
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = apply_logit_bias(logits.clone(), {0: 0.5, 2: -1.5})

    assert torch.allclose(out[0], torch.tensor(1.5))  # 1.0 + 0.5
    assert torch.allclose(out[1], torch.tensor(2.0))  # untouched
    assert torch.allclose(out[2], torch.tensor(1.5))  # 3.0 + (-1.5)
    assert torch.allclose(out[3], torch.tensor(4.0))  # untouched


def test_apply_logit_bias_overrides_penalties_via_addition():
    """Large positive bias can dominate natural penalties (additive semantics)."""
    logits = torch.tensor([1.0, 2.0])
    # Pretend a penalty subtracted 5.0 from token 0; logit_bias=10 restores it above token 1.
    out = apply_logit_bias(logits.clone(), {0: 10.0})
    assert torch.allclose(out[0], torch.tensor(11.0))  # 1.0 + 10.0
    assert torch.allclose(out[1], torch.tensor(2.0))   # untouched


def test_apply_logit_bias_does_not_mutate_input():
    """The helper returns a new tensor — input logits are unchanged."""
    logits = torch.tensor([1.0, 2.0, 3.0])
    snapshot = logits.clone()
    _ = apply_logit_bias(logits, {0: 5.0, 1: -2.0})
    assert torch.equal(logits, snapshot)


def test_apply_logit_bias_ignores_tokens_outside_vocab():
    """Token ids >= vocab_size are silently ignored (mirrors apply_presence_penalty)."""
    logits = torch.tensor([1.0, 2.0])
    out = apply_logit_bias(logits, {5: 100.0, 7: -50.0})
    # No id falls inside [0, vocab_size), so no bias applies.
    assert torch.equal(out, logits)


def test_apply_logit_bias_filters_partial_out_of_vocab():
    """Out-of-vocab ids are dropped; in-vocab ids still apply."""
    logits = torch.tensor([1.0, 2.0])
    out = apply_logit_bias(logits.clone(), {0: 0.5, 99: -10.0})
    # Token 99 is dropped; only token 0 shifts.
    assert torch.allclose(out[0], torch.tensor(1.5))
    assert torch.allclose(out[1], torch.tensor(2.0))


def test_apply_logit_bias_accepts_string_keys():
    """String keys (JSON boundary case) are coerced to int."""
    logits = torch.tensor([1.0, 2.0, 3.0])
    out = apply_logit_bias(logits.clone(), {"0": 0.5, "2": -1.0})
    assert torch.allclose(out[0], torch.tensor(1.5))
    assert torch.allclose(out[1], torch.tensor(2.0))
    assert torch.allclose(out[2], torch.tensor(2.0))


def test_apply_logit_bias_silently_drops_invalid_keys():
    """Keys that can't coerce to int are silently dropped."""
    logits = torch.tensor([1.0, 2.0])
    # "abc" is not parseable as int; should be dropped.
    out = apply_logit_bias(logits.clone(), {"abc": 99.0, 1: 0.5})
    assert torch.allclose(out[0], torch.tensor(1.0))  # abc dropped
    assert torch.allclose(out[1], torch.tensor(2.5))  # 1 + 0.5
