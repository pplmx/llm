"""Generation backend registry and bootstrap."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from llm.runtime.plugins import load_entry_point_registry
from llm.runtime.registry import Registry

if TYPE_CHECKING:
    from llm.generation.backends import GenerationBackend
    from llm.models.decoder import DecoderModel
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


def build_speculative_backend(
    *,
    target_model: DecoderModel | None = None,
    draft_model: DecoderModel | None = None,
    gamma: int = 5,
    **_kwargs: Any,
) -> GenerationBackend:
    """Build a speculative decoding backend (Leviathan et al., 2023).

    Both ``target_model`` and ``draft_model`` must share vocabulary
    with the tokenizer passed at generation time. The ``gamma``
    parameter controls how many candidate tokens the draft proposes
    per round.
    """
    from llm.generation.backends import SpeculativeDecodingBackend

    if target_model is None or draft_model is None:
        raise ValueError("speculative backend requires both target_model and draft_model kwargs")
    return SpeculativeDecodingBackend(
        target_model=target_model,
        draft_model=draft_model,
        gamma=gamma,
    )


def ensure_backends_registered() -> None:
    global _backends_registered
    if _backends_registered:
        return

    # Register the built-in speculative backend before loading entry
    # points so it always resolves even if no third-party plugin
    # provides one. The entry-point load raises if a plugin claims
    # the same name, which is intentional (the built-in is the
    # reference implementation).
    BACKEND_REGISTRY.register("speculative", build_speculative_backend)
    load_entry_point_registry("llm.generation_backends", BACKEND_REGISTRY)
    _backends_registered = True


def get_generation_backend(
    name: str = "eager",
    *,
    engine: ContinuousBatchingEngine | None = None,
    **kwargs: Any,
) -> GenerationBackend:
    """Resolve a generation backend by registry name.

    Backend-specific kwargs are forwarded to the factory — e.g.
    ``target_model=...``, ``draft_model=...``, ``gamma=...`` for the
    ``speculative`` backend, or ``engine=...`` for ``batched``.
    """
    ensure_backends_registered()
    return BACKEND_REGISTRY.get(name)(engine=engine, **kwargs)
