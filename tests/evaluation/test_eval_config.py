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
