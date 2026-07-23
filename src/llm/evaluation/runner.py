from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from llm.evaluation.eval_tasks.base import BaseTask


class EvaluationRunner:
    """Run evaluation tasks and persist reports."""

    def __init__(self, task: BaseTask, output_dir: str = "results"):
        self.task = task
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _collect_metrics(self, predictions: Any, references: Any) -> dict[str, Any]:
        """Run every task metric and return a flat ``{name: value}`` dict."""
        results: dict[str, Any] = {}
        for metric in self.task.metrics:
            results.update(metric.compute(predictions, references))
        return results

    def run(self, model, split: str = "test") -> dict:
        """Run evaluation with raw (non-tensor) predictions and references."""
        inputs, references = self.task.prepare_data(split)
        predictions = self.task.predict(model, inputs)

        results = {"num_samples": len(inputs)}
        results.update(self._collect_metrics(predictions, references))
        return results

    def evaluate(self, model, split: str = "val") -> dict:
        """Evaluate on a split, coercing tensor inputs when needed."""
        inputs, references = self.task.prepare_data(split)
        tensor_inputs = [torch.as_tensor(x, dtype=torch.long) for x in inputs]
        predictions = self.task.predict(model, tensor_inputs)

        refs = torch.stack([torch.as_tensor(x, dtype=torch.long) for x in references])
        return self._collect_metrics(predictions, refs)

    def save_report(self, results: dict, output_format: str = "json"):
        """Persist ``results`` to ``output_dir/eval_report.<ext>``.

        Args:
            results: Metric dict to serialise.
            output_format: One of ``"json"`` or ``"markdown"``.

        Raises:
            ValueError: if ``output_format`` is neither ``"json"`` nor
                ``"markdown"``.
        """
        if output_format == "json":
            output_file = self.output_dir / "eval_report.json"
            with output_file.open("w") as f:
                json.dump(results, f, indent=2)
        elif output_format == "markdown":
            output_file = self.output_dir / "eval_report.md"
            lines = ["# Evaluation Report\n"]
            for key, value in results.items():
                lines.append(f"- **{key}**: {value}")
            with output_file.open("w") as f:
                f.write("\n".join(lines))
        else:
            raise ValueError(f"unknown format '{output_format}'; supported: 'json', 'markdown'")
