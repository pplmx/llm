"""Generic plugin registry for runtime extensibility."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class Registry[T]:
    """Name-to-object registry with explicit registration and lookup."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, T] = {}

    def register(self, name: str, entry: T) -> T:
        if name in self._entries:
            raise ValueError(f"'{name}' is already registered in {self._name} registry.")
        self._entries[name] = entry
        return entry

    def get(self, name: str) -> T:
        if name not in self._entries:
            available = ", ".join(sorted(self._entries))
            raise ValueError(f"'{name}' not found in {self._name} registry. Available: {available}")
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries


def decorator_register(registry: Registry[type]) -> Callable[[str], Callable[[type], type]]:
    """Class decorator factory compatible with legacy register_model usage."""

    def register(name: str) -> Callable[[type], type]:
        def wrapper(cls: type) -> type:
            registry.register(name, cls)
            return cls

        return wrapper

    return register
