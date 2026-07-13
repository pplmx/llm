"""Benchmark presets for the lm-eval-harness pipeline.

A preset bundles a benchmark name with the lm_eval kwargs that
``evaluator.evaluate`` expects (``num_fewshot``, ``batch_size``,
``limit``, etc.). Three common presets ship out of the box; users
can extend by constructing :class:`EvalPreset` directly.

The presets are intentionally **decoupled from the lm_eval import**
so this module is safe to import on hosts that don't have
``lm_eval`` installed. Callers that actually want to *run* the
benchmark should use :func:`llm.evaluation.harness.adapter.run_preset`,
which is the boundary that imports lm_eval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalPreset:
    """A benchmark preset for the lm-eval-harness pipeline.

    Attributes:
        task: lm_eval task name (e.g. ``"mmlu"``, ``"arc_easy"``,
            ``"wikitext"``).
        num_fewshot: Number of few-shot exemplars. ``None`` lets
            lm_eval use the task's default.
        batch_size: Per-device evaluation batch size.
        limit: Optional cap on the number of samples per task
            (``None`` means run the full benchmark).
        task_kwargs: Extra kwargs forwarded to ``lm_eval.tasks``
            (e.g. ``{"subject": "abstract_algebra"}`` for a single
            MMLU subject).
        description: Human-readable one-liner for the report.
    """

    task: str
    num_fewshot: int | None = None
    batch_size: int = 8
    limit: int | None = None
    task_kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_lm_eval_kwargs(self) -> dict[str, Any]:
        """Flatten to the kwargs ``evaluator.evaluate`` understands.

        Always returns a fresh dict so callers can mutate it without
        poisoning the frozen preset.
        """
        kwargs: dict[str, Any] = {"tasks": [self.task], "batch_size": self.batch_size}
        if self.num_fewshot is not None:
            kwargs["num_fewshot"] = self.num_fewshot
        if self.limit is not None:
            kwargs["limit"] = self.limit
        if self.task_kwargs:
            kwargs["task_kwargs"] = dict(self.task_kwargs)
        return kwargs


# --- Built-in presets ------------------------------------------------------


MMLU_PRESET = EvalPreset(
    task="mmlu",
    num_fewshot=5,
    batch_size=8,
    description="Massive Multitask Language Understanding (5-shot).",
)


ARCEASY_PRESET = EvalPreset(
    task="arc_easy",
    num_fewshot=0,
    batch_size=8,
    description="AI2 Reasoning Challenge — easy split.",
)


WIKITEXT_PRESET = EvalPreset(
    task="wikitext",
    num_fewshot=0,
    batch_size=4,
    description="WikiText-103 perplexity (lm-eval's standard perplexity task).",
)


BUILTIN_PRESETS: dict[str, EvalPreset] = {
    MMLU_PRESET.task: MMLU_PRESET,
    ARCEASY_PRESET.task: ARCEASY_PRESET,
    WIKITEXT_PRESET.task: WIKITEXT_PRESET,
}


def get_preset(name: str) -> EvalPreset:
    """Look up a built-in preset by name.

    Raises:
        KeyError: if ``name`` is not one of the built-in presets.
    """
    if name not in BUILTIN_PRESETS:
        available = ", ".join(sorted(BUILTIN_PRESETS))
        raise KeyError(
            f"unknown preset {name!r}; available built-ins: {available}"
        )
    return BUILTIN_PRESETS[name]
