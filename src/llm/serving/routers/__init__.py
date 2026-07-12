"""FastAPI routers for the serving API.

Each router owns one logical endpoint group:

- ``health`` — liveness/readiness probes (``/health``).
- ``generate`` — text-generation endpoints (``/generate``,
  ``/batch_generate``).
- ``chat`` — OpenAI-compatible chat completions (``/v1/chat/completions``).
"""

from llm.serving.routers import chat, generate, health

__all__ = ["chat", "generate", "health"]
