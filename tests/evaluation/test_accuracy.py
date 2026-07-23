import pytest

from llm.evaluation.metrics.accuracy import AccuracyMetric, F1Metric

# AccuracyMetric is pure-Python (no optional deps); its tests must run in every
# environment. Only the F1 tests below are gated on scikit-learn.


def test_accuracy_basic():
    metric = AccuracyMetric()
    predictions = ["cat", "dog", "cat"]
    references = ["cat", "dog", "bird"]

    result = metric.compute(predictions, references)
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_accuracy_all_correct():
    metric = AccuracyMetric()
    predictions = ["a", "b", "c"]
    references = ["a", "b", "c"]

    result = metric.compute(predictions, references)
    assert result["accuracy"] == pytest.approx(1.0)


def test_accuracy_all_wrong():
    metric = AccuracyMetric()
    predictions = ["a", "b", "c"]
    references = ["x", "y", "z"]

    result = metric.compute(predictions, references)
    assert result["accuracy"] == pytest.approx(0.0)


def test_accuracy_empty():
    metric = AccuracyMetric()

    result = metric.compute([], [])
    assert result["accuracy"] == 0.0


def test_accuracy_mismatched_lengths_raise():
    metric = AccuracyMetric()
    predictions = ["a", "b"]
    references = ["a"]

    with pytest.raises(ValueError, match="zip\\(\\) argument"):
        metric.compute(predictions, references)


def test_accuracy_name():
    assert AccuracyMetric.name == "accuracy"


# --- F1 (requires scikit-learn, an optional [eval] dependency) ---


def test_f1_perfect():
    pytest.importorskip("sklearn", reason="scikit-learn is an optional eval dependency")
    metric = F1Metric()
    predictions = ["cat", "dog", "bird"]
    references = ["cat", "dog", "bird"]
    result = metric.compute(predictions, references)
    assert result["f1"] == pytest.approx(1.0)


def test_f1_partial():
    pytest.importorskip("sklearn", reason="scikit-learn is an optional eval dependency")
    metric = F1Metric()
    predictions = ["cat", "dog", "cat", "bird"]
    references = ["cat", "cat", "bird", "bird"]
    result = metric.compute(predictions, references)
    assert 0 < result["f1"] < 1


def test_f1_name():
    pytest.importorskip("sklearn", reason="scikit-learn is an optional eval dependency")
    assert F1Metric.name == "f1"
