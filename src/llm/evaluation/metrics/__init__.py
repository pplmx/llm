"""Metric implementations and the metric-name registry.

Metric classes may be instantiated directly with their constructor kwargs;
exposing them here also guarantees the side-effect module imports. To select
metrics BY NAME from configuration (``EvalConfig.metrics`` / RIL ISS-251)
use :func:`resolve_metrics`, which builds instances from
:data:`METRIC_REGISTRY` and raises on an unknown name instead of silently
running nothing.
"""

from llm.evaluation.metrics.accuracy import AccuracyMetric, F1Metric
from llm.evaluation.metrics.base import BaseMetric
from llm.evaluation.metrics.generation import BleuMetric, ChrFMetric, RougeMetric
from llm.evaluation.metrics.perplexity import PerplexityMetric

__all__ = [
    "METRIC_REGISTRY",
    "AccuracyMetric",
    "BaseMetric",
    "BleuMetric",
    "ChrFMetric",
    "F1Metric",
    "PerplexityMetric",
    "RougeMetric",
    "resolve_metrics",
]

#: Name -> metric class. Metric constructors are kwarg-free except the ones
#: with sensible defaults (``f1`` takes ``average="macro"``, ``perplexity``
#: takes ``ignore_index=-100``), so name-based resolution instantiates with
#: the class defaults — the per-task ``Task.metrics`` list remains the place
#: for task-specific constructor kwargs.
METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
    "accuracy": AccuracyMetric,
    "bleu": BleuMetric,
    "chrf": ChrFMetric,
    "f1": F1Metric,
    "perplexity": PerplexityMetric,
    "rouge": RougeMetric,
}


def resolve_metrics(names: list[str]) -> list[BaseMetric]:
    """Resolve metric NAMES to instances via :data:`METRIC_REGISTRY`.

    Raises:
        ValueError: naming a metric that is not registered (a loud fail —
            a typo'd ``EvalConfig.metrics`` must not silently run nothing,
            RIL ISS-251).
    """
    resolved: list[BaseMetric] = []
    for name in names:
        try:
            resolved.append(METRIC_REGISTRY[name]())
        except KeyError:
            available = ", ".join(sorted(METRIC_REGISTRY))
            raise ValueError(f"unknown metric {name!r}; available metrics: {available}") from None
    return resolved
