"""Health and readiness endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["health"])


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
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "llm",
                "object": "model",
                "owned_by": "llm",
            }
        ],
    }
