"""Backward-compatible re-exports for generation APIs.

Prefer ``llm.generation`` for new code.
"""

from llm.generation.eager import batch_generate, generate, stream_generate

__all__ = ["batch_generate", "generate", "stream_generate"]
