"""Generation subsystem: eager decoding and pluggable backends."""

from llm.generation.backends import (
    BatchedGenerationBackend,
    EagerGenerationBackend,
    GenerationBackend,
    GenerationConfig,
)
from llm.generation.eager import batch_generate, generate, stream_generate
from llm.generation.registry import (
    BACKEND_REGISTRY,
    build_batched_backend,
    build_eager_backend,
    ensure_backends_registered,
    get_generation_backend,
)

__all__ = [
    "BACKEND_REGISTRY",
    "BatchedGenerationBackend",
    "EagerGenerationBackend",
    "GenerationBackend",
    "GenerationConfig",
    "batch_generate",
    "build_batched_backend",
    "build_eager_backend",
    "ensure_backends_registered",
    "generate",
    "get_generation_backend",
    "stream_generate",
]
