"""Deprecated shim — use ``llm.training.task_registry`` instead."""

from llm.compat.legacy_imports import warn_deprecated
from llm.training.task_registry import TASK_REGISTRY, TaskRegistry, TaskSpec

warn_deprecated("llm.training.registry", "llm.training.task_registry")

__all__ = ["TASK_REGISTRY", "TaskRegistry", "TaskSpec"]
