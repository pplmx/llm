"""Deprecated shim — use ``llm.data.datasets.text`` instead."""

from llm.compat.legacy_imports import warn_deprecated
from llm.data.datasets.text import TextDataset, create_dataloader

warn_deprecated("llm.data.loader", "llm.data.datasets.text")

__all__ = ["TextDataset", "create_dataloader"]
