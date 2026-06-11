import json
from pathlib import Path

import torch

from llm.evaluation.eval_tasks.base import BaseTask


class EvaluationRunner:
    """Run evaluation tasks and persist reports."""

    def __init__(self, task: BaseTask, output_dir: str = "results"):
        self.task = task
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run(self, model, split: str = "test") -> dict:
        inputs, references = self.task.prepare_data(split)
        predictions = self.task.predict(model, inputs)

        results = {"num_samples": len(inputs)}
        for metric in self.task.metrics:
            metric_result = metric.compute(predictions, references)
            results.update(metric_result)

        return results

    def evaluate(self, model, split: str = "val") -> dict:
        """Evaluate on a split, coercing tensor inputs when needed."""
        inputs, references = self.task.prepare_data(split)
        tensor_inputs = [torch.as_tensor(x, dtype=torch.long) for x in inputs]
        predictions = self.task.predict(model, tensor_inputs)

        refs = torch.stack([torch.as_tensor(x, dtype=torch.long) for x in references])
        results = {}
        for metric in self.task.metrics:
            result = metric.compute(predictions, refs)
            results.update(result)
        return results

    def save_report(self, results: dict, format: str = "json"):
        output_file = self.output_dir / f"eval_report.{format}"

        if format == "json":
            with output_file.open("w") as f:
                json.dump(results, f, indent=2)
        elif format == "markdown":
            lines = ["# Evaluation Report\n"]
            for key, value in results.items():
                lines.append(f"- **{key}**: {value}")

            with output_file.open("w") as f:
                f.write("\n".join(lines))
