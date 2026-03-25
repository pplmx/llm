import pytest

from llm.evaluation.metrics.accuracy import AccuracyMetric, F1Metric


def test_accuracy_basic():
    metric = AccuracyMetric()
    predictions = ["cat", "dog", "cat"]
    references = ["cat", "dog", "bird"]

    result = metric.compute(predictions, references)
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_f1_basic():
    metric = F1Metric()
    predictions = ["cat", "dog", "cat", "bird"]
    references = ["cat", "cat", "bird", "bird"]

    result = metric.compute(predictions, references)
    assert "f1" in result


def test_f1_perfect():
    metric = F1Metric()
    predictions = ["cat", "dog", "bird"]
    references = ["cat", "dog", "bird"]
    result = metric.compute(predictions, references)
    assert result["f1"] == pytest.approx(1.0)


def test_f1_partial():
    metric = F1Metric()
    predictions = ["cat", "dog", "cat", "bird"]
    references = ["cat", "cat", "bird", "bird"]
    result = metric.compute(predictions, references)
    assert "f1" in result
    assert 0 < result["f1"] < 1
