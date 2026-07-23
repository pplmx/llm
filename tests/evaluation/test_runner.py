"""Tests for :class:`llm.evaluation.runner.EvaluationRunner`.

Covers ``save_report`` (JSON + markdown + unknown-format error) and the
``run`` method (which exercises the full predict -> metric loop with
plain Python predictions/references rather than tensor conversion).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from llm.evaluation.metrics.base import BaseMetric
from llm.evaluation.runner import EvaluationRunner

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _DummyMetric(BaseMetric):
    """Returns a fixed metric dict regardless of inputs."""

    name = "dummy"

    def compute(self, predictions, references):
        return {"dummy_score": 0.42}


class _StubTask:
    """Minimal stand-in for :class:`BaseTask`.

    ``prepare_data`` returns plain Python lists; ``predict`` echoes the
    inputs so tests can verify data flow without a real model.
    """

    name = "stub"
    metrics: ClassVar[list[BaseMetric]] = [_DummyMetric()]

    def prepare_data(self, split: str = "test"):
        return ["inp1", "inp2"], ["ref1", "ref2"]

    def predict(self, model, inputs):
        return [f"pred({i})" for i in inputs]


@pytest.fixture
def stub_runner(tmp_path: Path) -> EvaluationRunner:
    """An ``EvaluationRunner`` rooted in a temp output directory."""
    return EvaluationRunner(_StubTask(), output_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# run()
# --------------------------------------------------------------------------- #


def test_run_returns_num_samples_and_metric(stub_runner: EvaluationRunner):
    """``run`` returns ``num_samples`` plus per-metric results."""
    results = stub_runner.run(model=None)
    assert results["num_samples"] == 2
    assert results["dummy_score"] == pytest.approx(0.42)


# --------------------------------------------------------------------------- #
# save_report — JSON
# --------------------------------------------------------------------------- #


def test_save_report_json(stub_runner: EvaluationRunner):
    """JSON report is written and is parseable."""
    results = {"num_samples": 3, "perplexity": 12.5}
    stub_runner.save_report(results, output_format="json")
    report = stub_runner.output_dir / "eval_report.json"
    assert report.exists()
    loaded = json.loads(report.read_text())
    assert loaded == results


def test_save_report_json_indent(stub_runner: EvaluationRunner):
    """JSON report is indented for human readability."""
    stub_runner.save_report({"x": 1}, output_format="json")
    content = (stub_runner.output_dir / "eval_report.json").read_text()
    assert '"x"' in content
    # indent=2 means at least one newline after the opening brace
    assert content.startswith("{\n")


# --------------------------------------------------------------------------- #
# save_report — markdown
# --------------------------------------------------------------------------- #


def test_save_report_markdown(stub_runner: EvaluationRunner):
    """Markdown report uses ``- **key**: value`` lines and ``#`` heading."""
    results = {"num_samples": 3, "accuracy": 0.9}
    stub_runner.save_report(results, output_format="markdown")
    report = stub_runner.output_dir / "eval_report.md"
    assert report.exists()
    content = report.read_text()
    assert content.startswith("# Evaluation Report")
    assert "- **num_samples**: 3" in content
    assert "- **accuracy**: 0.9" in content


def test_save_report_markdown_extension_is_md(stub_runner: EvaluationRunner):
    """Markdown reports must use the conventional ``.md`` extension."""
    stub_runner.save_report({"x": 1}, output_format="markdown")
    assert (stub_runner.output_dir / "eval_report.md").exists()
    assert not (stub_runner.output_dir / "eval_report.markdown").exists()


# --------------------------------------------------------------------------- #
# save_report — error handling
# --------------------------------------------------------------------------- #


def test_save_report_unknown_format_raises(stub_runner: EvaluationRunner):
    """Unknown formats raise ``ValueError`` rather than silently no-op-ing."""
    with pytest.raises(ValueError, match="unknown format"):
        stub_runner.save_report({"x": 1}, output_format="yaml")


def test_save_report_unknown_format_does_not_write(stub_runner: EvaluationRunner):
    """An unknown format must not create a spurious file."""
    with pytest.raises(ValueError, match="unknown format"):
        stub_runner.save_report({"x": 1}, output_format="toml")
    assert not (stub_runner.output_dir / "eval_report.toml").exists()


# --------------------------------------------------------------------------- #
# default format
# --------------------------------------------------------------------------- #


def test_save_report_default_format_is_json(stub_runner: EvaluationRunner):
    """When no format is given, default to JSON."""
    stub_runner.save_report({"x": 1})
    assert (stub_runner.output_dir / "eval_report.json").exists()
