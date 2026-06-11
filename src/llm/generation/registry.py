"""Generation backend registry and bootstrap."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from llm.runtime.plugins import load_entry_point_registry
from llm.runtime.registry import Registry

if TYPE_CHECKING:
    from llm.generation.backends import GenerationBackend
    from llm.serving.batch_engine import ContinuousBatchingEngine

GenerationBackendFactory = Callable[..., "GenerationBackend"]

BACKEND_REGISTRY: Registry[GenerationBackendFactory] = Registry("GenerationBackend")

_backends_registered = False


def build_eager_backend(**_kwargs: Any) -> GenerationBackend:
    from llm.generation.backends import EagerGenerationBackend

    return EagerGenerationBackend()


def build_batched_backend(*, engine: ContinuousBatchingEngine | None = None, **_kwargs: Any) -> GenerationBackend:
    from llm.generation.backends import BatchedGenerationBackend

    if engine is None:
        raise ValueError("batched generation backend requires a ContinuousBatchingEngine instance")
    return BatchedGenerationBackend(engine)


def ensure_backends_registered() -> None:
    global _backends_registered
    if _backends_registered:
        return

    load_entry_point_registry("llm.generation_backends", BACKEND_REGISTRY)
    _backends_registered = True


def get_generation_backend(
    name: str = "eager",
    *,
    engine: ContinuousBatchingEngine | None = None,
) -> GenerationBackend:
    """Resolve a generation backend by registry name."""
    ensure_backends_registered()
    return BACKEND_REGISTRY.get(name)(engine=engine)
