"""PerplexityMetric behaviors — registry default, narrow references, shape.

Covers the deep-dive findings:

: the metric-name registry resolves ``perplexity`` with
  ``ignore_index=None`` while ``metrics/__init__.py`` documents ``-100``;
  with the documented default an all-ignored corpus would return NaN
  instead of the contracted ``inf``.
: references *narrower* than predictions crashed with a raw
  batch-mismatch ValueError, while only the predictions-narrower direction
  was clamped.
"""

import pytest
import torch

from llm.evaluation.metrics import resolve_metrics
from llm.evaluation.metrics.perplexity import PerplexityMetric


def test_registry_resolves_perplexity_with_ignore_index_100():
    """Name-resolved ``perplexity`` must default to ``ignore_index=-100`` (the
    value documented in ``metrics/__init__.py``), not ``None``.
    With ``None`` the metric scores pad/ignored positions and an
    all-ignored corpus yields NaN instead of the documented ``inf``."""
    metric = resolve_metrics(["perplexity"])[0]
    assert isinstance(metric, PerplexityMetric)
    assert metric.ignore_index == -100, (
        f"registry perplexity resolved ignore_index={metric.ignore_index!r}; metrics/__init__.py documents -100"
    )


def test_all_ignored_corpus_returns_inf():
    """With ignore_index=-100, a corpus whose targets are all ignored must
    return the documented ``inf`` (perplexity undefined), never NaN."""
    preds = torch.zeros(1, 4, 10)
    refs = torch.full((1, 4), -100, dtype=torch.long)
    result = PerplexityMetric(ignore_index=-100).compute(preds, refs)
    assert result["perplexity"] == float("inf")
    assert result["perplexity"] != float("nan")


def test_references_narrower_than_predictions_does_not_crash():
    """References with a shorter sequence axis than predictions must be scored
    (the narrower tail has no logits) instead of raising a raw batch-mismatch
    ValueError (the fix only clamped the
    predictions-narrower direction)."""
    pm = PerplexityMetric()
    preds = torch.randn(2, 5, 10)  # seq_len 5
    refs = torch.ones(2, 3, dtype=torch.long)  # seq_len 3 — narrower
    result = pm.compute(preds, refs)
    ppl = result["perplexity"]
    assert ppl > 0
    assert ppl != float("inf")


def test_references_wider_than_predictions_still_scores():
    """The direction (references wider than predictions) stays
    clamped: the extra reference positions were never scored."""
    pm = PerplexityMetric()
    preds = torch.randn(2, 3, 10)  # seq_len 3
    refs = torch.ones(2, 5, dtype=torch.long)  # seq_len 5 — wider
    result = pm.compute(preds, refs)
    ppl = result["perplexity"]
    assert ppl > 0


def test_empty_prediction_list_returns_inf():
    """An empty eval split must report the documented ``inf``.

    ``PerplexityMetric.compute`` documents ``inf`` for an empty batch and
    ``LMTask.predict`` already returns a zero-size tensor for an empty
    split — but a caller passing plain lists (``compute([], [])``, or a task
    whose ``predict`` returns raw rows) hit a bare
    ``AttributeError: 'list' object has no attribute 'shape'`` on the very
    first line that touched ``predictions.shape``. Every sibling metric
    reports its zero value on empty input, so this path is the odd one out.
    """
    assert PerplexityMetric().compute([], []) == {"perplexity": float("inf")}


def test_non_tensor_predictions_raise_actionable_type_error():
    """A non-empty non-tensor prediction list must fail with a message that
    names the expected shape, not an opaque ``AttributeError`` on ``.shape``."""
    pm = PerplexityMetric()
    with pytest.raises(TypeError, match=r"\(batch, seq, vocab\)"):
        pm.compute([1, 2, 3], [1, 2, 3])


# --- the alignment window must be the RIGHT window ------------------------
#
# The tests above assert only that each direction "does not crash" and returns
# a positive number, which any value satisfies. They cannot tell a correct
# clamp from a misaligned one: slicing the logits one position off, or scoring
# the wrong number of positions, still yields a positive finite perplexity.
# These compute the expected value independently.


def test_the_clamped_window_scores_exactly_the_shift_targets_that_exist():
    """Values must equal a hand-computed shifted CE over the shared window."""
    torch.manual_seed(1)
    batch, seq, vocab = 1, 6, 10
    logits = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, seq))

    got = PerplexityMetric().compute(logits, [labels.squeeze(0)])["perplexity"]

    # Reference: shift by one on both sides, mean CE, then exp.
    shifted_logits = logits[:, :-1, :].reshape(-1, vocab)
    shifted_labels = labels[:, 1:].reshape(-1)
    expected = float(torch.exp(torch.nn.functional.cross_entropy(shifted_logits, shifted_labels)))

    assert abs(got - expected) < 1e-5, f"perplexity {got} != independently computed {expected}"


def test_a_narrower_reference_scores_the_positions_both_sides_share():
    """With refs narrower than logits, the value is the CE over ref_len-1 steps.

    A misaligned clamp (off by one, or scoring a different span) produces a
    different finite number, so comparing the VALUE is what catches it.
    """
    torch.manual_seed(2)
    batch, seq, vocab = 2, 6, 10
    logits = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, 4))  # narrower: 4 < 6

    got = PerplexityMetric().compute(logits, labels)["perplexity"]

    # label_width = min(ref_len - 1, seq - 1) = 3, start-aligned on both sides.
    window = 3
    expected = float(
        torch.exp(
            torch.nn.functional.cross_entropy(
                logits[:, :window, :].reshape(-1, vocab),
                labels[:, 1 : 1 + window].reshape(-1),
            )
        )
    )

    assert abs(got - expected) < 1e-5, f"perplexity {got} != independently computed {expected}"


def test_a_wider_reference_ignores_only_the_unscoreable_tail():
    """With refs wider than logits, the value is the CE over seq-1 steps."""
    torch.manual_seed(3)
    batch, seq, vocab = 2, 4, 10
    logits = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, 8))  # wider: 8 > 4

    got = PerplexityMetric().compute(logits, labels)["perplexity"]

    window = seq - 1  # 3
    expected = float(
        torch.exp(
            torch.nn.functional.cross_entropy(
                logits[:, :window, :].reshape(-1, vocab),
                labels[:, 1 : 1 + window].reshape(-1),
            )
        )
    )

    assert abs(got - expected) < 1e-5, f"perplexity {got} != independently computed {expected}"
