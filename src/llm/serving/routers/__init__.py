"""FastAPI routers for the serving API.

Each router owns one logical endpoint group:

- ``health`` — liveness/readiness probes (``/health``, ``/v1/models``).
- ``generate`` — text-generation endpoints (``/generate``,
  ``/batch_generate``).
- ``chat`` — OpenAI-compatible chat completions (``/v1/chat/completions``).
- ``completions`` — OpenAI-compatible legacy text completions
  (``/v1/completions``).
"""

from llm.serving.routers import chat, completions, generate, health

__all__ = ["chat", "completions", "generate", "health"]
