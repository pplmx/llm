"""Tests for :mod:`llm.evaluation.config.EvalConfig`.

Coverage was 0 % previously — this pins down the pydantic defaults
and the ability to override them.
"""

from __future__ import annotations

from llm.evaluation.config import EvalConfig


def test_eval_config_defaults():
    """Default EvalConfig has expected values."""
    config = EvalConfig()
    assert config.eval_interval == 1000
    assert config.eval_batch_size == 8
    assert config.metrics == ["perplexity", "accuracy"]


def test_eval_config_custom_values():
    """EvalConfig accepts overrides."""
    config = EvalConfig(
        eval_interval=500,
        eval_batch_size=16,
        metrics=["f1", "rouge"],
    )
    assert config.eval_interval == 500
    assert config.eval_batch_size == 16
    assert config.metrics == ["f1", "rouge"]


def test_eval_config_eval_interval_positive():
    """eval_interval can be set to any positive integer."""
    config = EvalConfig(eval_interval=1)
    assert config.eval_interval == 1


def test_eval_config_metrics_names_resolve():
    """``EvalConfig.metrics`` names resolve to real metric instances.

    The names were previously DEAD CONFIG — nothing looked them up, so
    setting ``metrics=['rouge', 'f1']`` ran whatever ``task.metrics``
    happened to be. After ISS-251 the names go through the metric registry.
    """
    from llm.evaluation.metrics import METRIC_REGISTRY, resolve_metrics

    resolved = {metric.name for metric in resolve_metrics(EvalConfig().metrics)}
    assert {"perplexity", "accuracy"} <= resolved
    for name in ("perplexity", "accuracy", "f1", "rouge", "bleu", "chrf"):
        assert name in METRIC_REGISTRY


def test_eval_config_unknown_metric_name_raises_not_silent():
    """A typo'd metric name fails loudly instead of silently running nothing."""
    import pytest

    from llm.evaluation.metrics import resolve_metrics

    with pytest.raises(ValueError, match="unknown metric"):
        resolve_metrics(["not_a_real_metric"])
