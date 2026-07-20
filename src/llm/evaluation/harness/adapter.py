"""Adapter for lm-evaluation-harness.

Two responsibilities:

1. ``LmEvalAdapter`` — thin wrapper around ``lm_eval.evaluator`` that
   adds structured result handling and preset support.
2. ``run_preset`` — convenience that ties a preset, an LM, and the
   evaluator together.

The :mod:`llm.evaluation.harness.presets` module is safe to import
without ``lm_eval`` installed; the lm_eval import boundary lives
here so the project's existing ``[eval]`` optional-dependency group
keeps working.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from llm.evaluation.harness.presets import EvalPreset, get_preset

# Soft-dependency probe — mirrors the convention used elsewhere
# (flash-attn, huggingface-hub, safetensors) so CI on a CPU-only host
# doesn't crash at import time.
_lm_eval_spec = importlib.util.find_spec("lm_eval")
LM_EVAL_AVAILABLE: bool = _lm_eval_spec is not None


def _require_lm_eval() -> None:
    """Raise a clear ``ImportError`` with the install command."""
    if not LM_EVAL_AVAILABLE:
        raise ImportError(
            "lm-eval integration requires the optional 'lm_eval' package. Install with `pip install 'llm[eval]'`."
        )


class LmEvalAdapter:
    """Adapter for lm-evaluation-harness.

    Adds:

    * **Preset lookup** — ``run_preset(preset_name, lm)`` resolves a
      :class:`EvalPreset` by name (built-in or user-supplied) and
      runs the benchmark.
    * **Structured result flattening** — :meth:`summarize` extracts
      ``acc`` / ``acc_norm`` / ``perplexity`` / ``f1`` / etc. from
      the nested lm_eval result shape into a flat
      ``{task_name: {metric: value}}`` dict.
    """

    def __init__(self) -> None:
        _require_lm_eval()
        from lm_eval.tasks import TaskManager

        self._task_manager = TaskManager()

    def list_tasks(self) -> list[str]:
        """List available benchmark tasks."""
        return sorted(self._task_manager.all_tasks)

    def evaluate(self, model: Any, tasks: list[str] | None = None, **kwargs: Any) -> dict:
        """Run evaluation on specified tasks.

        Mirrors the lm_eval ``evaluator.evaluate`` signature; see
        ``lm_eval`` docs for the kwargs surface.
        """
        _require_lm_eval()
        from lm_eval import evaluator

        return evaluator.evaluate(model=model, tasks=tasks or ["mmlu"], **kwargs)

    def run_preset(
        self,
        preset: EvalPreset | str,
        model: Any,
        **kwargs: Any,
    ) -> dict:
        """Run a benchmark by preset (name or :class:`EvalPreset`).

        Merges the preset's :meth:`EvalPreset.to_lm_eval_kwargs` with
        any caller-supplied ``kwargs`` (caller wins on conflicts).
        """
        if isinstance(preset, str):
            preset = get_preset(preset)
        merged = preset.to_lm_eval_kwargs()
        merged.update(kwargs)
        return self.evaluate(model, **merged)

    def run_benchmark(self, model: Any, benchmark: str, **kwargs: Any) -> dict:
        """Run a single benchmark task by name (no preset lookup)."""
        return self.evaluate(model, tasks=[benchmark], **kwargs)

    @staticmethod
    def summarize(results: dict[str, Any]) -> dict[str, dict[str, float]]:
        """Flatten lm_eval's nested result tree into a flat metric map.

        lm_eval's ``evaluator.simple_evaluate`` returns:

        .. code-block:: python

            {
                "results": {
                    "task_name": {
                        "acc,none": 0.42,
                        "acc_norm,none": 0.45,
                        ...
                    },
                    ...
                },
                "groups": {...},
                "configs": {...},
            }

        This helper extracts the ``results`` block and splits each
        comma-separated key (``"acc,none"`` -> metric ``"acc"``,
        subset ``"none"``) so callers can serialize it as a flat
        dict.

        Notes:

        - Only the ``results`` block is flattened; ``groups`` and
          ``configs`` are intentionally ignored.
        - Booleans are dropped (Python's ``bool`` is a subclass of
          ``int``, but a metric of value ``True`` is almost certainly
          a bug — string aliases are usually what you'd see).
        - Non-numeric values (e.g. string aliases) are silently
          skipped, keeping the output strictly numeric.
        """
        flat: dict[str, dict[str, float]] = {}
        for task_name, metrics in results.get("results", {}).items():
            flat[task_name] = {}
            for key, value in metrics.items():
                if "," in key:
                    metric_name = key.split(",", 1)[0]
                else:
                    metric_name = key
                if isinstance(value, bool):
                    # bool is technically int — skip to avoid ``True`` -> 1.0.
                    continue
                if isinstance(value, (int, float)):
                    flat[task_name][metric_name] = float(value)
        return flat


def run_preset(
    preset: EvalPreset | str,
    model: Any,
    **kwargs: Any,
) -> dict:
    """Convenience entry point: build an adapter and run a preset."""
    return LmEvalAdapter().run_preset(preset, model, **kwargs)
