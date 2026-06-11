"""Bridge legacy model registry API to runtime ModelFactory."""

from typing import Any

from llm.runtime import MODEL_REGISTRY as _RUNTIME_MODEL_REGISTRY
from llm.runtime.registry import decorator_register

# Backward-compatible dict view used by older imports.
MODEL_REGISTRY: dict[str, type[Any]] = _RUNTIME_MODEL_REGISTRY._entries  # type: ignore[assignment]

register_model = decorator_register(_RUNTIME_MODEL_REGISTRY)


def get_model(name: str) -> type[Any]:
    """Retrieve a registered model builder by name."""
    return _RUNTIME_MODEL_REGISTRY.get(name)
