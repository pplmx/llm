"""Backwards-compatibility shim for the checkpoint surface.

The checkpoint implementation moved to :mod:`llm.checkpoint` because it was
never really a *training* concern: ``serving``, ``runtime`` and ``cli`` read
checkpoints without training anything, yet had to import the training layer to
do it.

- :mod:`llm.checkpoint.format` — the v2 split layout, legacy detection, atomic
  writes, and legacy-to-v2 migration (pure file I/O).
- :mod:`llm.checkpoint.manager` — the training-side ``CheckpointManager``
  (retention policy, best-loss tracking, epoch snapshots).
- :mod:`llm.checkpoint.errors` — the exception types both sides raise.

This module re-exports the whole surface so existing imports keep working. New
code should import from :mod:`llm.checkpoint` directly.
"""

from __future__ import annotations

from llm.checkpoint.errors import CheckpointIncompleteError, CheckpointMigrationError
from llm.checkpoint.format import (
    CHECKPOINT_FORMAT_VERSION,
    EXTRA_STATE_SUFFIX,
    LEGACY_SUFFIX,
    META_SUFFIX,
    SAFETENSORS_SUFFIX,
    convert_legacy_checkpoint_to_split,
    load_checkpoint_payload,
)
from llm.checkpoint.manager import CheckpointManager

# ``CheckpointConfig`` was implicitly importable from this module before the
# split (it was a top-level import here). Keep that working.
from llm.training.core.config import CheckpointConfig

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "EXTRA_STATE_SUFFIX",
    "LEGACY_SUFFIX",
    "META_SUFFIX",
    "SAFETENSORS_SUFFIX",
    "CheckpointConfig",
    "CheckpointIncompleteError",
    "CheckpointManager",
    "CheckpointMigrationError",
    "convert_legacy_checkpoint_to_split",
    "load_checkpoint_payload",
]
