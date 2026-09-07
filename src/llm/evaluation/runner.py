from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from llm.evaluation.eval_tasks.base import BaseTask


def _to_serializable(obj: Any) -> Any:
    """Recursively convert tensors and numpy scalars to JSON-safe Python types.

    Walks lists, tuples, and dicts.  ``torch.Tensor`` values are reduced to
    their scalar via ``.item()``; numpy scalars via ``.tolist()``.  ``nan``
    and ``inf`` are normalised to ``None`` so the emitted JSON is strictly
    valid (the default ``json`` encoder would otherwise emit the
    non-standard tokens ``NaN`` / ``Infinity``).
    """
    # torch.Tensor → scalar or list
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return _to_serializable(obj.item())
        return _to_serializable(obj.tolist())
    # numpy scalar → Python native
    if hasattr(obj, "tolist"):
        try:
            converted = obj.tolist()
        except TypeError:
            return obj
        except ValueError:
            return obj
        # tolist() on a 0-d array returns a Python scalar; on 1-d returns a list
        return _to_serializable(converted)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _as_token_tensor(value: Any) -> Any:
    """Coerce a token row to a ``long`` tensor, passing text through.

    ``evaluate`` promises tensor inputs, so numeric rows (int lists,
    tensors) are coerced like before; a STRING row (generation task
    references) is left as-is — ``as_tensor(str, dtype=torch.long)`` raises
    an opaque ``TypeError`` the ``run`` path never hit (RIL ISS-396).
    """
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, str):
        return value
    try:
        return torch.as_tensor(value, dtype=torch.long)
    except TypeError, ValueError:
        # Uncoercible rows (e.g. ragged nested lists) stay raw; the metric
        # layer either consumes them or raises its own clearer error.
        return value


def _align_reference_device(predictions: Any, references: Any) -> Any:
    """Move ``references`` onto the ``predictions`` device.

    A model evaluated where it lives (CUDA) yields CUDA predictions while
    ``prepare_data`` returns CPU tensors; feeding CPU references straight to
    ``nll_loss`` then raises a device mismatch (RIL TASK-329's periodic
    training eval hit exactly this). Non-tensor references (e.g. strings for
    generation metrics) pass through unchanged.
    """
    if not isinstance(predictions, torch.Tensor):
        return references
    device = predictions.device
    if isinstance(references, torch.Tensor):
        return references.to(device)
    if isinstance(references, list):
        return [r.to(device) if isinstance(r, torch.Tensor) else r for r in references]
    return references


class EvaluationRunner:
    """Run evaluation tasks and persist reports.

    Args:
        task: The evaluation task (carries ``task.metrics`` — metric
            instances with task-specific constructor kwargs).
        output_dir: Where reports are written.
        metric_names: Optional metric NAMES resolved via
            :func:`llm.evaluation.metrics.resolve_metrics` (RIL ISS-251).
            When ``None`` or empty, the runner falls back to
            ``task.metrics``.
    """

    def __init__(self, task: BaseTask, output_dir: str = "results", metric_names: list[str] | None = None):
        self.task = task
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if metric_names:
            from llm.evaluation.metrics import resolve_metrics

            self.metrics = resolve_metrics(metric_names)
        else:
            self.metrics = list(task.metrics)

    def _collect_metrics(self, predictions: Any, references: Any) -> dict[str, Any]:
        """Run every configured metric and return a flat ``{name: value}`` dict."""
        results: dict[str, Any] = {}
        for metric in self.metrics:
            results.update(metric.compute(predictions, references))
        return results

    def run(self, model, split: str = "test") -> dict:
        """Run evaluation with raw (non-tensor) predictions and references."""
        inputs, references = self.task.prepare_data(split)
        predictions = self.task.predict(model, inputs)
        references = _align_reference_device(predictions, references)

        results = {"num_samples": len(inputs)}
        results.update(self._collect_metrics(predictions, references))
        return results

    def evaluate(self, model, split: str = "val") -> dict:
        """Evaluate on a split, coercing tensor inputs when needed.

        Mirrors :meth:`run`'s result shape (both report ``num_samples``) and
        accepts BOTH numeric (tensor-coercible) and string references: an
        ``as_tensor(..., dtype=torch.long)`` coercion of a generation task's
        text rows raised an opaque ``TypeError`` while :meth:`run` served
        the same task fine (RIL ISS-396). Non-numeric rows pass through
        unchanged; numeric ones still reach the metrics as tensors —
        PerplexityMetric stacks lists of tensors, so the two-row stack form
        is preserved.
        """
        inputs, references = self.task.prepare_data(split)
        tensor_inputs = [_as_token_tensor(x) for x in inputs]
        predictions = self.task.predict(model, tensor_inputs)

        if references:
            # ``run`` passes raw (possibly string) references straight to the
            # metrics; ``evaluate``'s docs promise tensor inputs, so coerce
            # what is coercible and leave text rows alone (generation
            # metrics consume them as-is). All-tensor references stack to
            # the (batch, seq) tensor the perplexity path expects.
            coerced = [_as_token_tensor(x) for x in references]
            refs = torch.stack(coerced) if coerced and all(isinstance(r, torch.Tensor) for r in coerced) else coerced
        else:
            # Empty eval set: mirror the empty prediction so the metric
            # layer sees a zero-size batch and reports ``inf`` rather than
            # crashing on ``torch.stack([])``.
            refs = torch.empty(0, dtype=torch.long)

        results = {"num_samples": len(inputs)}
        results.update(self._collect_metrics(predictions, _align_reference_device(predictions, refs)))
        return results

    def save_report(self, results: dict, output_format: str = "json"):
        """Persist ``results`` to ``output_dir/eval_report.<ext>``.

        Args:
            results: Metric dict to serialise.
            output_format: One of ``"json"`` or ``"markdown"``.

        Raises:
            ValueError: if ``output_format`` is neither ``"json"`` nor
                ``"markdown"``.
        """
        safe_results = _to_serializable(results)
        if output_format == "json":
            output_file = self.output_dir / "eval_report.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(safe_results, f, indent=2)
        elif output_format == "markdown":
            output_file = self.output_dir / "eval_report.md"
            lines = ["# Evaluation Report\n"]
            for key, value in safe_results.items():
                lines.append(f"- **{key}**: {value}")
            with output_file.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        else:
            raise ValueError(f"unknown format '{output_format}'; supported: 'json', 'markdown'")
