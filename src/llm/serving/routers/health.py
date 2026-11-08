"""Health and readiness endpoints."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["health"])

#: Creation timestamp reported by ``GET /v1/models``.
#:
#: OpenAI's ``Model`` object requires ``created`` (an epoch-seconds int), and
#: the official SDK's pydantic model **rejects a body without it** —
#: ``client.models.list()`` raised ``ValidationError: created: Field required``
#: against this server. That call is the first thing many clients make, so the
#: failure surfaced as "server unreachable" before any completion was tried,
#: which is exactly the symptom the endpoint was added to prevent.
#:
#: The value is the process start time rather than a per-request ``now()``: the
#: model "exists" from when the server loaded it, and a stable timestamp keeps
#: repeated probes from looking like the model is being re-created.
_SERVER_STARTED_AT: int = int(time.time())


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe. Returns ``{"status": "ok"}`` if the process is up."""
    return {"status": "ok"}


# OpenAI clients (including the official SDK's ``client.models.list()``) call
# ``GET /v1/models`` to discover what the server can serve. Its absence made
# that call fail with a bare 404, which some clients surface as "server
# unreachable" before they ever send a completion. The server hosts exactly one
# model, so this returns a single-entry list in the documented shape.
@router.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """List the served model (OpenAI ``/v1/models`` shape).

    ``/v1/chat/completions`` ignores the request's ``model`` field — the server
    serves whatever was loaded at startup — so the id here is informational. It
    still has to exist: clients use the endpoint as a reachability probe.

    Every field the SDK's ``Model`` type requires is emitted (``id``,
    ``created``, ``object``, ``owned_by``). Omitting ``created`` looked correct
    against OpenAI's prose docs and still broke the real client, which is why
    ``tests/serving/test_openai_sdk_compat.py`` validates this response against
    that model rather than against a hand-written expected dict.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "llm",
                "object": "model",
                "created": _SERVER_STARTED_AT,
                "owned_by": "llm",
            }
        ],
    }
