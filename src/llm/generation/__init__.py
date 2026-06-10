"""Generation subsystem: eager decoding and pluggable backends."""

from llm.generation.backends import EagerGenerationBackend, GenerationBackend, GenerationConfig, get_generation_backend
from llm.generation.eager import batch_generate, generate, stream_generate

__all__ = [
    "EagerGenerationBackend",
    "GenerationBackend",
    "GenerationConfig",
    "batch_generate",
    "generate",
    "get_generation_backend",
    "stream_generate",
]
