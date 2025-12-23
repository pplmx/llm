from typing import Any, TypeVar

T = TypeVar("T")


class ComponentRegistry:
    """
    A simple registry for model components (Attention, MLP, Norm, etc.).
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: dict[str, Any] = {}

    def register(self, name: str) -> Any:
        def decorator(cls: T) -> T:
            if name in self._registry:
                raise ValueError(f"Component '{name}' already registered in {self._name} registry.")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Any:
        if name not in self._registry:
            raise ValueError(
                f"Component '{name}' not found in {self._name} registry. Available: {list(self._registry.keys())}"
            )
        return self._registry[name]


# Global registries
ATTENTION_REGISTRY = ComponentRegistry("Attention")
MLP_REGISTRY = ComponentRegistry("MLP")
NORM_REGISTRY = ComponentRegistry("Normalization")
