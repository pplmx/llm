"""Discover third-party plugins via setuptools entry points."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from llm.runtime.registry import Registry

logger = logging.getLogger(__name__)


def _iter_group_entry_points(group: str):
    eps = entry_points()
    if hasattr(eps, "select"):
        return eps.select(group=group)
    return eps.get(group, [])


def _load_one(name: str, group: str, ep):
    """Load and call an entry point; swallow + log a broken plugin.

    A third-party plugin whose module fails to import (a missing dependency,
    a syntax error, an unguarded import) must not take down the rest of the
    registry — the built-ins and every healthy plugin still register. Before
    this, one broken ``llm.models`` entry point aborted the WHOLE loop, so
    ``import llm.runtime`` (which runs ``ensure_builtins_registered``) failed
    outright and nothing — including the built-ins — could be used
    (RIL ISS-131).
    """
    try:
        return ep.load()
    except Exception as exc:  # noqa: BLE001 - one bad plugin must not block others
        logger.error(
            "Failed to load plugin entry point '%s' in group '%s': %s: %s",
            name,
            group,
            type(exc).__name__,
            exc,
        )
        return None


def load_entry_point_registry[T](
    group: str,
    registry: Registry[T],
    *,
    overwrite: bool = False,
) -> list[str]:
    """Load callables from entry points into a registry.

    Returns names that were newly registered. A plugin that fails to load is
    logged (not fatal) — the rest of the group still registers (RIL ISS-131).
    """
    loaded: list[str] = []
    for ep in _iter_group_entry_points(group):
        preexisting = ep.name in registry
        if not overwrite and preexisting:
            continue
        factory = _load_one(ep.name, group, ep)
        if factory is None:
            continue
        if preexisting:
            # ``overwrite=True`` and the name is already registered:
            # ``Registry.register`` would raise, so use the explicit
            # replace path (RIL ISS-061).
            registry.replace(ep.name, factory)
        else:
            registry.register(ep.name, factory)
        loaded.append(ep.name)
    return loaded


def load_entry_point_hooks(group: str) -> list[str]:
    """Invoke zero-arg registration hooks from entry points.

    Hooks that fail to load or raise are logged, not fatal — the remaining
    hooks still run (RIL ISS-131).
    """
    invoked: list[str] = []
    for ep in _iter_group_entry_points(group):
        hook = _load_one(ep.name, group, ep)
        if hook is None:
            continue
        if not callable(hook):
            logger.error(
                "Entry point '%s' in group '%s' is not callable; skipped.",
                ep.name,
                group,
            )
            continue
        try:
            hook()
        except Exception as exc:  # noqa: BLE001 - one bad hook must not block others
            logger.error(
                "Hook '%s' in group '%s' raised %s: %s",
                ep.name,
                group,
                type(exc).__name__,
                exc,
            )
            continue
        invoked.append(ep.name)
    return invoked
