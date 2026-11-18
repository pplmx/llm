from importlib import import_module

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
    # Patch the symbol the metric actually calls. Hooking ``builtins.__import__``
    # looked equivalent but was not: ``importlib.import_module`` returns a
    # module straight from ``sys.modules`` when it is already imported, so once
    # scikit-learn IS installed (``uv sync --extra eval``, which the eval suite
    # is meant to run under) the hook never executes, the metric succeeds, and
    # this test fails — demanding the dependency be absent to test the
    # missing-dependency message. Patching ``import_module`` in the metric's
    # own namespace exercises the same code path regardless of what is
    # installed.
    from llm.evaluation.metrics import accuracy as accuracy_module

    def _fake_import_module(name: str, *args, **kwargs):
        if name == "sklearn.metrics" or name.startswith("sklearn"):
            raise ImportError("No module named 'sklearn'")
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(accuracy_module, "import_module", _fake_import_module)
    with pytest.raises(ImportError, match=r"llm\[eval\]"):
        F1Metric().compute(["a"], ["b"])


# --- wrong argument types must say so ---------------------------------------
#
# ``BaseMetric.compute`` declares ``predictions: Any``, and the sibling
# ``PerplexityMetric`` genuinely takes tensors, so a caller reaching for
# ``accuracy`` with a tensor is making a reasonable mistake. Before the guard,
# that raised "Boolean value of Tensor with more than one value is ambiguous" —
# an error naming neither the argument nor the fix.


def test_a_tensor_argument_is_rejected_with_an_actionable_message():
    """A tensor must fail with a TypeError that says what to do instead."""
    import torch

    from llm.evaluation.metrics.accuracy import AccuracyMetric

    with pytest.raises(TypeError) as excinfo:
        AccuracyMetric().compute(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([0, 1]))

    message = str(excinfo.value)
    assert "torch.Tensor" in message
    assert "tolist" in message, "the error should name the fix, not just the problem"


def test_a_tensor_reference_is_rejected_too():
    """The guard covers both arguments, not just predictions."""
    import torch

    from llm.evaluation.metrics.accuracy import AccuracyMetric

    with pytest.raises(TypeError, match="references"):
        AccuracyMetric().compute([0, 1], torch.tensor([0, 1]))


def test_plain_sequences_still_work():
    """The guard must not reject the documented list input."""
    from llm.evaluation.metrics.accuracy import AccuracyMetric

    metric = AccuracyMetric()
    assert metric.compute([0, 1, 2, 1], [0, 1, 0, 1])["accuracy"] == 0.75
    assert metric.compute([], [])["accuracy"] == 0.0


def test_f1_rejects_a_tensor_with_the_same_message():
    """``F1Metric`` had the identical defect and needs the identical guard."""
    import torch

    from llm.evaluation.metrics.accuracy import F1Metric

    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        F1Metric().compute(torch.tensor([0, 1]), torch.tensor([0, 1]))
