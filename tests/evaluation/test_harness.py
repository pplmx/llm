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


class TestSummarizeFlatteningRules:
    """The documented rules for ``summarize`` each need a test.

    It is a pure static method (works without ``lm_eval`` installed), so every
    rule in its docstring is directly checkable — and each exists to prevent a
    specific misreading of lm_eval's result tree.
    """

    def test_comma_suffix_is_the_aggregation_tag_not_a_subset(self):
        """``"acc,none"`` collapses to ``"acc"``; different metric names survive.

        lm_eval's suffix is the FILTER tag (almost always ``none`` for these
        tasks), not a subset name — grouped benchmarks report their subsets as
        separate top-level task keys. Dropping the tag is therefore lossless, and
        a regression that dropped the metric PREFIX instead would turn every
        task into a single ``"none"`` entry.
        """
        from llm.evaluation.harness.adapter import LmEvalAdapter

        flat = LmEvalAdapter.summarize(
            {"results": {"arc_easy": {"acc,none": 0.75, "acc_norm,none": 0.70, "acc_stderr,none": 0.01}}}
        )
        assert flat == {"arc_easy": {"acc": 0.75, "acc_norm": 0.70, "acc_stderr": 0.01}}

    def test_booleans_are_dropped(self):
        """``True`` must not become ``1.0``.

        Python's ``bool`` is an ``int`` subclass, so without the explicit check a
        boolean flag in the tree would be coerced into a plausible-looking
        metric value. The docstring calls this out; nothing was testing it.
        """
        from llm.evaluation.harness.adapter import LmEvalAdapter

        flat = LmEvalAdapter.summarize(
            {
                "results": {
                    "task": {
                        "acc,none": 0.5,
                        "is_broken,none": True,
                        "is_ok,none": False,
                    }
                }
            }
        )
        assert flat == {"task": {"acc": 0.5}}, f"booleans leaked into the metrics: {flat}"

    def test_non_numeric_values_are_skipped(self):
        """String aliases/versions are not metrics."""
        from llm.evaluation.harness.adapter import LmEvalAdapter

        flat = LmEvalAdapter.summarize(
            {"results": {"task": {"acc,none": 0.5, "alias": "arc_easy", "version,none": "1.0"}}}
        )
        assert flat == {"task": {"acc": 0.5}}

    def test_groups_and_configs_are_ignored(self):
        """Only the ``results`` block is flattened (documented behaviour).

        Callers who want the group roll-ups must read the raw dict; silently
        merging them here would overwrite per-task numbers with aggregates.
        """
        from llm.evaluation.harness.adapter import LmEvalAdapter

        flat = LmEvalAdapter.summarize(
            {
                "results": {"arc_easy": {"acc,none": 0.75}},
                "groups": {"arc": {"acc,none": 0.99}},
                "configs": {"arc_easy": {"num_fewshot": 0}},
            }
        )
        assert flat == {"arc_easy": {"acc": 0.75}}, f"group/config blocks leaked in: {flat}"

    def test_empty_and_missing_results_blocks(self):
        """A run with no results yields an empty map, not a crash."""
        from llm.evaluation.harness.adapter import LmEvalAdapter

        assert LmEvalAdapter.summarize({"results": {}}) == {}
        assert LmEvalAdapter.summarize({}) == {}
