import sys

import pytest

from llm.evaluation.metrics.generation import BleuMetric, ChrFMetric, RougeMetric


def test_rouge_basic():
    pytest.importorskip("rouge_score", reason="rouge_score is an optional eval dependency")
    pytest.importorskip("nltk", reason="nltk is an optional eval dependency (rouge stemmer)")
    metric = RougeMetric()
    predictions = ["the cat sat on the mat"]
    references = ["the cat sat on the rug"]

    result = metric.compute(predictions, references)
    assert result["rouge-l"] > 0.5


def test_rouge_custom_types():
    pytest.importorskip("rouge_score", reason="rouge_score is an optional eval dependency")
    pytest.importorskip("nltk", reason="nltk is an optional eval dependency (rouge stemmer)")
    metric = RougeMetric(rouge_types=["rouge1"])
    predictions = ["a b c"]
    references = ["a b c"]

    result = metric.compute(predictions, references)
    assert "rouge-1" in result
    assert result["rouge-1"] == pytest.approx(1.0)


def test_bleu_basic():
    pytest.importorskip("sacrebleu", reason="sacrebleu is an optional eval dependency")
    metric = BleuMetric()
    predictions = ["the cat sat on the mat"]
    references = ["the cat sat on the mat"]

    result = metric.compute(predictions, references)
    assert result["bleu"] == pytest.approx(100.0, rel=0.1)


def test_bleu_partial_overlap():
    pytest.importorskip("sacrebleu", reason="sacrebleu is an optional eval dependency")
    metric = BleuMetric()
    predictions = ["the cat sat on the mat"]
    references = ["the dog sat on the rug"]

    result = metric.compute(predictions, references)
    assert 0 < result["bleu"] <= 100.0


def test_chrf_basic():
    pytest.importorskip("sacrebleu", reason="sacrebleu is an optional eval dependency")
    metric = ChrFMetric()
    predictions = ["the cat sat on the mat"]
    references = ["the cat sat on the mat"]

    result = metric.compute(predictions, references)
    assert "chrf" in result


def test_chrf_no_overlap():
    pytest.importorskip("sacrebleu", reason="sacrebleu is an optional eval dependency")
    metric = ChrFMetric()
    predictions = ["cat"]
    references = ["dog"]

    result = metric.compute(predictions, references)
    assert "chrf" in result
    assert result["chrf"] < 30.0


# --------------------------------------------------------------------------- #
# Soft-dependency contract for RougeMetric
#
# ``RougeMetric`` must be constructible without ``rouge-score`` installed
# (same contract as ``BleuMetric`` / ``ChrFMetric`` which import sacrebleu
# lazily inside ``compute``). We monkey-patch ``sys.modules`` so the test
# runs in *every* environment — with or without rouge_score present.
# --------------------------------------------------------------------------- #


def test_rouge_metric_constructible_without_rouge_score(monkeypatch):
    """``RougeMetric()`` must not import ``rouge_score`` eagerly."""
    monkeypatch.setitem(sys.modules, "rouge_score", None)

    metric = RougeMetric()
    assert metric.rouge_types == ["rouge1", "rouge2", "rougeL"]


def test_rouge_metric_custom_types_without_rouge_score(monkeypatch):
    """Custom ``rouge_types`` are stored at init without triggering import."""
    monkeypatch.setitem(sys.modules, "rouge_score", None)

    metric = RougeMetric(rouge_types=["rouge1"])
    assert metric.rouge_types == ["rouge1"]


def test_rouge_metric_compute_raises_without_rouge_score(monkeypatch):
    """``compute`` on a host without rouge_score raises a clear ImportError."""
    monkeypatch.setitem(sys.modules, "rouge_score", None)

    metric = RougeMetric()
    with pytest.raises(ImportError, match="rouge-score"):
        metric.compute(["pred"], ["ref"])


def test_rouge_metric_empty_inputs_without_rouge_score(monkeypatch):
    """Empty predictions/references return ``{}`` without importing rouge_score."""
    monkeypatch.setitem(sys.modules, "rouge_score", None)

    metric = RougeMetric()
    result = metric.compute([], [])
    assert result == {}


# --------------------------------------------------------------------------- #
# Soft-dependency contract for BleuMetric / ChrFMetric
#
# Both must be constructible without ``sacrebleu`` installed, and their
# ``compute`` must raise a clear ``ImportError`` (not a bare ModuleNotFoundError)
# when sacrebleu is missing — same contract as RougeMetric above.
# --------------------------------------------------------------------------- #


def test_bleu_constructible_without_sacrebleu(monkeypatch):
    """BleuMetric() must not import sacrebleu eagerly."""
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    metric = BleuMetric()
    assert metric.name == "bleu"


def test_bleu_compute_raises_without_sacrebleu(monkeypatch):
    """compute without sacrebleu raises a clear ImportError with install hint."""
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    metric = BleuMetric()
    with pytest.raises(ImportError, match=r"pip install 'llm\[eval\]'"):
        metric.compute(["pred"], ["ref"])


def test_chrf_constructible_without_sacrebleu(monkeypatch):
    """ChrFMetric() must not import sacrebleu eagerly."""
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    metric = ChrFMetric()
    assert metric.name == "chrf"


def test_chrf_compute_raises_without_sacrebleu(monkeypatch):
    """compute without sacrebleu raises a clear ImportError with install hint."""
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    metric = ChrFMetric()
    with pytest.raises(ImportError, match=r"pip install 'llm\[eval\]'"):
        metric.compute(["pred"], ["ref"])
