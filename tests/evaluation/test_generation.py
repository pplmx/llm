import pytest

from llm.evaluation.metrics.generation import BleuMetric, ChrFMetric, RougeMetric

# Each metric depends on its own optional package. Gate the import *per test*
# so that e.g. BleuMetric/ChrFMetric run whenever sacrebleu is available,
# independent of whether rouge_score (used only by RougeMetric) is installed.


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
