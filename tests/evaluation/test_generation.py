import pytest

from llm.evaluation.metrics.generation import BleuMetric, ChrFMetric, RougeMetric


def test_rouge_basic():
    metric = RougeMetric()
    predictions = ["the cat sat on the mat"]
    references = ["the cat sat on the rug"]

    result = metric.compute(predictions, references)
    assert "rouge-1" in result or "rouge-l" in result


def test_bleu_basic():
    metric = BleuMetric()
    predictions = ["the cat sat on the mat"]
    references = ["the cat sat on the mat"]

    result = metric.compute(predictions, references)
    assert result["bleu"] == pytest.approx(100.0, rel=0.1)


def test_chrf_basic():
    metric = ChrFMetric()
    predictions = ["the cat sat on the mat"]
    references = ["the cat sat on the mat"]

    result = metric.compute(predictions, references)
    assert "chrf" in result
