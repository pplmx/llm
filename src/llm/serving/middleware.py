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
