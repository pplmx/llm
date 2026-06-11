"""Checkpoint contributor protocol and helpers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CheckpointContributor(Protocol):
    """Component that persists auxiliary training state in checkpoint extra_state."""

    def get_checkpoint_state(self) -> dict[str, Any] | None:
        """Return a partial extra_state fragment to merge into the checkpoint."""

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore state from the merged checkpoint extra_state."""


def _is_contributor(obj: Any) -> bool:
    return isinstance(obj, CheckpointContributor)


def collect_extra_state(*contributors: Any) -> dict[str, Any] | None:
    """Merge extra_state fragments from all checkpoint contributors."""
    merged: dict[str, Any] = {}
    for contributor in contributors:
        if not _is_contributor(contributor):
            continue
        fragment = contributor.get_checkpoint_state()
        if fragment:
            merged.update(fragment)
    return merged or None


def load_extra_state(state: dict[str, Any] | None, *contributors: Any) -> None:
    """Restore auxiliary state into all checkpoint contributors."""
    if not state:
        return
    for contributor in contributors:
        if _is_contributor(contributor):
            contributor.load_checkpoint_state(state)
