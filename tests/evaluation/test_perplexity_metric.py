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
