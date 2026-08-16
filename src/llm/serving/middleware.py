"""ASGI middleware for the serving API.

Right now this holds the :class:`RequestIDMiddleware`, which assigns a
stable ``X-Request-ID`` to every request (honoring an inbound header),
echoes it on the response, and logs a structured access line per request
so operators can correlate uvicorn access logs with application logs.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class _RequestBodyTooLargeError(Exception):
    """Internal sentinel: request body exceeded the configured cap."""


class RequestBodySizeLimit:
    """Pure-ASGI middleware rejecting request bodies over ``max_bytes``.

    Defense-in-depth for client memory-exhaustion DoS (RIL ISS-171): without
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

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

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
                return await send(self._too_large_response())

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
            await send(self._too_large_response())
            return

    @staticmethod
    def _too_large_response() -> dict:
        """An ASGI ``send`` single-call payload (413 with a JSON envelope)."""
        body = b'{"error":{"message":"Request body too large","type":"invalid_request_error"}}'
        return {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }


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
