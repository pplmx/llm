"""Data layer: datasets, modules, and streaming sources."""

from llm.data.base import BaseDataModule, MapDataModule, StreamDataModule
from llm.data.presets import (
    BUILTIN_PRESETS,
    C4_PRESET,
    DatasetPreset,
    REDPAJAMA_PRESETS,
    THEPILE_PRESET,
    apply_to_config,
    list_presets,
    resolve_preset,
)

__all__ = [
    "BaseDataModule",
    "MapDataModule",
    "StreamDataModule",
    "DatasetPreset",
    "C4_PRESET",
    "THEPILE_PRESET",
    "REDPAJAMA_PRESETS",
    "BUILTIN_PRESETS",
    "apply_to_config",
    "resolve_preset",
    "list_presets",
]
