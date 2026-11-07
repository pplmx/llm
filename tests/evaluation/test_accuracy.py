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
    """Length mismatch must name the contract, not the ``zip`` mechanism.

    ``zip(..., strict=True)`` raised "zip() argument 2 is shorter than
    argument 1", which tells a task author nothing about the actual mistake
    (``predict()`` returned a different number of rows than ``prepare_data``).
    """
    metric = AccuracyMetric()
    predictions = ["a", "b"]
    references = ["a"]

    with pytest.raises(ValueError, match="one reference per prediction"):
        metric.compute(predictions, references)


def test_accuracy_mismatch_error_names_both_counts():
    """The message must report both lengths so the skewed side is obvious."""
    with pytest.raises(ValueError, match=r"3 predictions and 2 references"):
        AccuracyMetric().compute(["a", "b", "c"], ["a", "b"])


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


def test_f1_empty_returns_zero():
    """F1Metric must not crash on empty inputs — return 0.0, like AccuracyMetric."""
    pytest.importorskip("sklearn", reason="scikit-learn is an optional eval dependency")
    metric = F1Metric()
    result = metric.compute([], [])
    assert result == {"f1": 0.0}


def test_f1_missing_sklearn_raises_actionable_install_hint(monkeypatch):
    """A missing ``scikit-learn`` must name the extra to install.

    ``F1Metric`` let the raw ``ModuleNotFoundError: No module named
    'sklearn'`` escape, while every sibling soft-dependency metric
    (bleu / chrf / rouge) converts it into an install hint. The message named
    neither the extra nor the package, so a user had to read the source to
    learn what to run.
    """
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("No module named 'sklearn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError, match=r"llm\[eval\]"):
        F1Metric().compute(["a"], ["b"])
