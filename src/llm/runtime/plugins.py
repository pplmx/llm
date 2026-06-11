"""Discover third-party plugins via setuptools entry points."""

from __future__ import annotations

from importlib.metadata import entry_points

from llm.runtime.registry import Registry


def _iter_group_entry_points(group: str):
    eps = entry_points()
    if hasattr(eps, "select"):
        return eps.select(group=group)
    return eps.get(group, [])


def load_entry_point_registry[T](
    group: str,
    registry: Registry[T],
    *,
    overwrite: bool = False,
) -> list[str]:
    """Load callables from entry points into a registry.

    Returns names that were newly registered.
    """
    loaded: list[str] = []
    for ep in _iter_group_entry_points(group):
        if not overwrite and ep.name in registry:
            continue
        factory = ep.load()
        registry.register(ep.name, factory)
        loaded.append(ep.name)
    return loaded


def load_entry_point_hooks(group: str) -> list[str]:
    """Invoke zero-arg registration hooks from entry points."""
    invoked: list[str] = []
    for ep in _iter_group_entry_points(group):
        hook = ep.load()
        if not callable(hook):
            raise TypeError(f"Entry point '{ep.name}' in group '{group}' must be callable.")
        hook()
        invoked.append(ep.name)
    return invoked
