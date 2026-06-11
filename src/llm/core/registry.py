"""Component registries backed by runtime.Registry."""

from __future__ import annotations

from typing import Any, TypeVar

from llm.runtime.registry import Registry

T = TypeVar("T")


class ComponentRegistry:
    """Decorator-style registry for core model components."""

    def __init__(self, name: str) -> None:
        self._registry: Registry[Any] = Registry(name)

    def register(self, name: str):
        def decorator(cls: T) -> T:
            self._registry.register(name, cls)
            return cls

        return decorator

    def get(self, name: str) -> Any:
        return self._registry.get(name)

    def names(self) -> list[str]:
        return self._registry.names()

    def __contains__(self, name: str) -> bool:
        return name in self._registry


ATTENTION_REGISTRY = ComponentRegistry("Attention")
MLP_REGISTRY = ComponentRegistry("MLP")
NORM_REGISTRY = ComponentRegistry("Normalization")
