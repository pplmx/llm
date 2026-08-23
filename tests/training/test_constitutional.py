"""Tests for the rule-based Constitutional-AI slice (TASK-235 / ROADMAP 11.4)."""

from __future__ import annotations

import torch

from llm.training.rlhf.constitutional import (
    Constitution,
    EndsWithToken,
    ForbiddenToken,
    constitutional_loop,
    critique,
    revise,
)


def test_forbidden_token_principle():
    principle = ForbiddenToken({7})
    assert principle.is_satisfied([1, 2, 3])
    assert not principle.is_satisfied([1, 7, 3])
    assert not principle.is_satisfied([7])


def test_ends_with_token_principle():
    principle = EndsWithToken(1)
    assert principle.is_satisfied([5, 4, 1])
    assert not principle.is_satisfied([5, 4, 2])
    assert not principle.is_satisfied([])  # empty response cannot end with the target


def test_constitution_score_and_violations():
    constitution = Constitution([ForbiddenToken({7}), EndsWithToken(1)])
    # Satisfies neither forbidden-token (has 7) nor end-with (ends in 3).
    assert constitution.score([7, 2, 3]) == 0.0
    assert len(constitution.violations([7, 2, 3])) == 2
    # Satisfies both.
    assert constitution.score([2, 3, 1]) == 1.0


def test_critique_reports_violations():
    constitution = Constitution([ForbiddenToken({7}), EndsWithToken(1)])
    text = critique([7, 2, 3], constitution)
    assert "forbid-tokens" in text
    assert "end-with-token" in text
    assert critique([2, 3, 1], constitution) == "satisfies all principles"


def test_revise_makes_response_comply():
    constitution = Constitution([ForbiddenToken({7}), EndsWithToken(1)])
    revised = revise([7, 2, 3], constitution, safe_token=0)
    expected = [0, 2, 1]  # forbidden 7 -> safe 0; last token forced to target 1
    assert revised == expected
    assert constitution.satisfied_count(revised) == len(constitution.principles)


def test_constitutional_loop_rejects_wrong_shape():
    constitution = Constitution([ForbiddenToken({7})])
    import pytest

    with pytest.raises(ValueError, match="B, L"):
        constitutional_loop(torch.zeros(4, 3, 2), constitution)


def test_constitutional_e2e_scores_improve():
    """CPU e2e: critique + revision raises the fraction of constitution
    principles satisfied — revised responses comply fully and score strictly
    higher than originals that violated the constitution."""
    constitution = Constitution([ForbiddenToken({7}), EndsWithToken(1)])
    gen = torch.Generator()
    gen.manual_seed(0)
    responses = torch.randint(1, 16, (32, 8), generator=gen)

    result = constitutional_loop(responses, constitution, safe_token=0)
    before = result["scores_before"]
    after = result["scores_after"]

    assert before.mean() < 1.0, "synthetic responses must not all comply initially"
    assert (after == 1.0).all(), "revision must make every response fully comply"
    assert after.mean() > before.mean()
    # Every originally-violating row must strictly improve.
    improved = after[before < 1.0]
    assert bool((improved > before[before < 1.0]).all())

    # Critique text non-empty and length matches the batch.
    assert len(result["critiques"]) == 32
    assert all(isinstance(c, str) and c for c in result["critiques"])
    # Revisions are valid token lists of the same length as the input rows.
    assert len(result["revisions"]) == 32
    assert all(len(r) == responses.size(1) for r in result["revisions"])
