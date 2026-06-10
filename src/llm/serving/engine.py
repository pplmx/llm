"""Deprecated shim — use ``llm.serving.batch_engine`` instead."""

from llm.compat.legacy_imports import warn_deprecated
from llm.serving.batch_engine import ContinuousBatchingEngine, SlotAllocator

warn_deprecated("llm.serving.engine", "llm.serving.batch_engine")

__all__ = ["ContinuousBatchingEngine", "SlotAllocator"]
