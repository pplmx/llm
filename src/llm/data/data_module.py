"""Deprecated shim — use ``llm.data.base`` instead."""

from llm.compat.legacy_imports import warn_deprecated
from llm.data.base import BaseDataModule, MapDataModule, StreamDataModule

warn_deprecated("llm.data.data_module", "llm.data.base")

__all__ = ["BaseDataModule", "MapDataModule", "StreamDataModule"]
