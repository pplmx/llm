"""Checkpoint-specific exception types.

Split out so both the format layer and the training-side manager can raise the
same types without importing each other.
"""

from __future__ import annotations


class CheckpointIncompleteError(RuntimeError):
    """A checkpoint's sidecar files are present but do not form a saved trio.

    Raised when the guard markers (``save_id`` / ``epoch``) stamped into the
    sidecars disagree, which means the trio came from different saves or a
    crash tore it in half. Callers must not silently fall back to an earlier
    epoch here: the risk is training from stale optimizer state while looking
    like a successful resume.
    """


class CheckpointMigrationError(RuntimeError):
    """A legacy-to-v2 checkpoint conversion could not complete."""
