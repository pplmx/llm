"""Checkpoint file format: v2 split layout, read/write, migration.

This package owns the **on-disk representation** of a checkpoint and knows
nothing about training runs. It used to live inside
``training/core/checkpoint.py``, which meant `serving`, `runtime` and `cli` —
none of which train anything — had to import the training layer just to read a
file off disk.

Two modules:

- :mod:`llm.checkpoint.format` — the v2 split layout (safetensors weights +
  JSON metadata + pickled training-state sidecar), legacy single-file
  detection, atomic writes, and legacy-to-v2 migration. Pure file I/O plus
  payload-shape validation; it names fields like ``optimizer_state`` because
  that is what the format contains, not because it interprets them.
- :mod:`llm.checkpoint.manager` — the training-side :class:`CheckpointManager`
  (retention policy, best-loss tracking, epoch snapshots). Imported by the
  training layer and by nothing below it.

``llm.training.core.checkpoint`` re-exports the whole surface, so existing
imports keep working.
"""

from llm.checkpoint.format import (
    CHECKPOINT_FORMAT_VERSION,
    EXTRA_STATE_SUFFIX,
    LEGACY_SUFFIX,
    META_SUFFIX,
    SAFETENSORS_SUFFIX,
    CheckpointIncompleteError,
    CheckpointMigrationError,
    convert_legacy_checkpoint_to_split,
    load_checkpoint_payload,
)
from llm.checkpoint.manager import CheckpointManager

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "EXTRA_STATE_SUFFIX",
    "LEGACY_SUFFIX",
    "META_SUFFIX",
    "SAFETENSORS_SUFFIX",
    "CheckpointIncompleteError",
    "CheckpointManager",
    "CheckpointMigrationError",
    "convert_legacy_checkpoint_to_split",
    "load_checkpoint_payload",
]
