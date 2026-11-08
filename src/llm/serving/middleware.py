"""ASGI middleware for the serving API.

Right now this holds the :class:`RequestIDMiddleware`, which assigns a
stable ``X-Request-ID`` to every request (honoring an inbound header),
echoes it on the response, and logs a structured access line per request
so operators can correlate uvicorn access logs with application logs.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from typing import cast

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class _RequestBodyTooLargeError(Exception):
    """Internal sentinel: request body exceeded the configured cap."""


class RequestBodySizeLimit:
    """Pure-ASGI middleware rejecting request bodies over ``max_bytes``.

    Defense-in-depth for client memory-exhaustion DoS: without
    a limit, a single multi-hundred-MB JSON body (one giant ``prompt``, many
    chat messages) is fully buffered and tokenized. Two layers here:

    - **Content-Length fast reject** — when the client declares a length
      over the cap, respond ``413 Payload Too Large`` before reading any body.
    - **Incremental cap for chunked / unknown-length bodies** — counts bytes
      as the ASGI ``http.request`` messages stream in and rejects as soon as
      the cap is crossed, discarding (never buffering) the remainder.

    Implemented as raw ASGI (not :class:`BaseHTTPMiddleware`) so it wraps the
    transport layer and does NOT interfere with SSE response streaming.
    """

    def __init__(self, app, max_bytes: int | Callable[[], int]):
        self.app = app
        # A callable defers the cap to request time. The app registers this
        # middleware at import, while ``ServingConfig`` can still be replaced
        # by an entry point (``apply_config``) right up to startup; a frozen
        # int would keep whatever the environment held at import.
        if callable(max_bytes):
            # ``cast`` because ``int`` is itself callable by the type checker's
            # reckoning (it has a ``__call__``-ish signature through the ABCs),
            # so the ``callable()`` narrowing cannot prove the branch.
            self._resolve_max_bytes: Callable[[], int] = cast("Callable[[], int]", max_bytes)
        else:
            fixed = max_bytes
            self._resolve_max_bytes = lambda: fixed

    @property
    def max_bytes(self) -> int:
        """The effective cap in bytes (re-resolved on every request)."""
        return self._resolve_max_bytes()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except TypeError, ValueError:
                declared = None
            if declared is not None and declared > self.max_bytes:
                return await self._send_too_large(scope, send)

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body") or b"")
                if received > self.max_bytes:
                    # Drain the remainder so the connection is reusable, then
                    # abort the request; the bytes are counted, not buffered.
                    while message.get("more_body", False):
                        message = await receive()
                    raise _RequestBodyTooLargeError()
            return message

        try:
            return await self.app(scope, limited_receive, send)
        except _RequestBodyTooLargeError:
            await self._send_too_large(scope, send)
            return

    async def _send_too_large(self, scope, send) -> None:
        """Emit the complete 413 response — ``start`` AND ``body``.

        Sending only ``http.response.start`` with a ``content-length`` header
        is an ASGI protocol violation: uvicorn logs "ASGI callable returned
        without completing response" and closes the connection before writing
        the body, so a client that was promised N bytes receives none and
        reports a truncated transfer. Measured with curl:
        ``curl: (18) transfer closed with 77 bytes remaining to read``. The
        HTTP client sees a framing error instead of the 413 it was sent, which
        is strictly worse than no size cap at all (the request "fails" in a
        way that looks like a network fault).

        The body also now matches the documented error envelope
        (:func:`llm.serving.errors.to_envelope`): the old payload used
        OpenAI's ``type``/``message`` spelling, so a client parsing
        ``error.code`` — the shape every other error on this API uses — found
        nothing.
        """
        # Reuse the shared code/status source of truth rather than repeating
        # the literals, so a client branching on ErrorCode.PAYLOAD_TOO_LARGE and
        # this payload cannot drift apart.
        from llm.serving.errors import ErrorCode, default_status_for

        body = json.dumps(
            {
                "error": {
                    "code": ErrorCode.PAYLOAD_TOO_LARGE.value,
                    "message": f"Request body too large (the limit is {self.max_bytes} bytes).",
                    "details": {"limit_bytes": self.max_bytes},
                    "request_id": _request_id_from_scope(scope),
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": default_status_for(ErrorCode.PAYLOAD_TOO_LARGE),
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _request_id_from_scope(scope: dict) -> str:
    """The request id assigned by :class:`RequestIDMiddleware`, if any.

    That middleware runs *outside* this one, so by the time the size cap fires
    it has already stashed the id on ``scope["state"]`` (Starlette shares one
    state dict between the two, and the id is set before the inner app is
    called). Echoing it keeps the 413 correlatable with its access-log line
    instead of being the one error on this API with no request id.
    """
    state = scope.get("state") or {}
    return state.get("request_id", "unknown")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign, propagate, and log ``X-Request-ID`` for every request.

    Behavior:
    - If the client sent ``X-Request-ID``, reuse it (so callers can stitch
      retries to a single trace).
    - Otherwise, generate a new UUID4 hex.
    - Store on ``request.state.request_id`` so handlers and exception
      handlers can include it in error envelopes.
    - Echo on the response ``X-Request-ID`` header.
    - Log a structured INFO line on response (method, path, status,
      duration_ms, request_id).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        inbound = request.headers.get(REQUEST_ID_HEADER)
        request_id = inbound if inbound else uuid.uuid4().hex
        request.state.request_id = request_id

        start = time.perf_counter()
        status_code = 500  # default if call_next raises before returning
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "request",
                extra={
                    "event": "request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )
