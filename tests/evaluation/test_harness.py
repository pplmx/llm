"""Tests for the lm-eval-harness adapter soft-dependency contract.

The ``[eval]`` group (lm_eval) is optional, so the adapter must degrade to a
clear ``ImportError`` when it is absent. Those two tests below run in *every*
environment (with and without lm_eval) by monkey-patching the module-level
``LM_EVAL_AVAILABLE`` flag. The one test that needs the real lm_eval
``TaskManager`` is gated with a per-test ``importorskip`` so it only skips the
test that actually requires the dependency.
"""

from __future__ import annotations

import pytest

from llm.evaluation.harness.adapter import LmEvalAdapter, run_preset


def test_adapter_raises_importerror_when_lm_eval_missing(monkeypatch):
    """``LmEvalAdapter()`` raises a clear, actionable ``ImportError``."""
    import llm.evaluation.harness.adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "LM_EVAL_AVAILABLE", False)

    with pytest.raises(ImportError, match="lm-eval integration requires"):
        LmEvalAdapter()


def test_module_run_preset_raises_importerror_when_missing(monkeypatch):
    """The module-level :func:`run_preset` helper surfaces the same guard."""
    import llm.evaluation.harness.adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "LM_EVAL_AVAILABLE", False)

    with pytest.raises(ImportError, match="lm-eval integration requires"):
        run_preset("mmlu", model=None)


def test_adapter_summarize_is_pure_python():
    """``summarize`` is a static method that must work without lm_eval."""
    out = LmEvalAdapter.summarize({"results": {"mmlu": {"acc,none": 0.42}}})
    assert out == {"mmlu": {"acc": 0.42}}


def test_lm_eval_adapter_lists_known_benchmarks():
    pytest.importorskip("lm_eval", reason="lm_eval is an optional eval dependency")
    adapter = LmEvalAdapter()
    tasks = adapter.list_tasks()
    assert "mmlu" in tasks or "arc" in tasks
