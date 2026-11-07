from importlib import import_module

from llm.evaluation.metrics.base import BaseMetric


class AccuracyMetric(BaseMetric):
    """Accuracy metric for classification tasks."""

    name = "accuracy"

    def compute(self, predictions: list, references: list) -> dict:
        """Compute accuracy."""
        if len(predictions) != len(references):
            # A task whose ``predict`` returns a different number of rows than
            # ``prepare_data`` is a contract violation upstream, but the bare
            # ``zip(..., strict=True)`` surfaced it as
            # "zip() argument 2 is longer than argument 1" — which names the
            # mechanism, not the mistake. Say which side is short.
            raise ValueError(
                f"accuracy needs one reference per prediction, got "
                f"{len(predictions)} predictions and {len(references)} references; "
                "a task's predict() must return exactly as many rows as prepare_data()."
            )
        correct = sum(p == r for p, r in zip(predictions, references, strict=True))
        acc = correct / len(predictions) if predictions else 0.0
        return {"accuracy": acc}


class F1Metric(BaseMetric):
    """F1 score metric for classification tasks.

    Requires the ``scikit-learn`` package, available via the ``[eval]`` extra.
    """

    name = "f1"

    def __init__(self, average: str = "macro"):
        self.average = average

    def compute(self, predictions: list, references: list) -> dict:
        """Compute F1 score using sklearn.

        Returns ``{"f1": 0.0}`` for empty inputs, matching
        :meth:`AccuracyMetric.compute`'s convention.
        """
        if not predictions:
            return {"f1": 0.0}

        # ``scikit-learn`` is an optional extra. Every sibling soft-dependency
        # metric (bleu / chrf / rouge) converts the raw ImportError into an
        # actionable install hint; F1 let ``ModuleNotFoundError: No module
        # named 'sklearn'`` escape, which names neither the extra nor the
        # package the user should install.
        try:
            sklearn_metrics = import_module("sklearn.metrics")
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is an optional evaluation dependency. Install with `pip install 'llm[eval]'`."
            ) from exc
        f1 = sklearn_metrics.f1_score(references, predictions, average=self.average, zero_division=0)
        return {"f1": f1}
