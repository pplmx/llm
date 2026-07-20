"""Data layer: datasets, modules, and streaming sources."""

from llm.data.base import BaseDataModule, MapDataModule, StreamDataModule
from llm.data.presets import (
    BUILTIN_PRESETS,
    C4_PRESET,
    REDPAJAMA_PRESETS,
    THEPILE_PRESET,
    DatasetPreset,
    apply_to_config,
    list_presets,
    resolve_preset,
)

__all__ = [
    "BUILTIN_PRESETS",
    "C4_PRESET",
    "REDPAJAMA_PRESETS",
    "THEPILE_PRESET",
    "BaseDataModule",
    "DatasetPreset",
    "MapDataModule",
    "StreamDataModule",
    "apply_to_config",
    "list_presets",
    "resolve_preset",
]
