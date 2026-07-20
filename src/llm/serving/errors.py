"""Structured error envelope for the serving API (Finding K).

All endpoints in ``src/llm/serving`` raise ``APIError`` (or an HTTPException
that the global handler maps to an envelope). The envelope shape is::

    {
      "error": {
        "code": "<stable_id>",   # machine-readable, e.g. "invalid_request"
        "message": "<human>",    # one-line description
        "details": {...},        # structured context (field-level errors, etc.)
        "request_id": "<uuid>"   # X-Request-ID echo
      }
    }

The HTTP status is set by the ``status_code`` field on ``APIError`` (or by
mapping ``ErrorCode`` to its default status). The X-Request-ID middleware
sets ``request.state.request_id`` and echoes it on the response header.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ErrorCode(StrEnum):
    """Stable machine-readable error identifiers.

    These strings are part of the public API. Adding new codes is fine;
    renaming or removing them is a breaking change.
    """

    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED = "unauthorized"
    TIMEOUT = "timeout"
    MODEL_UNAVAILABLE = "model_unavailable"
    INTERNAL = "internal"


# Default HTTP status for each code. Clients can rely on these to branch
# (e.g. retry on 503, refresh token on 403).
_CODE_TO_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.UNAUTHORIZED: 403,
    ErrorCode.TIMEOUT: 504,
    ErrorCode.MODEL_UNAVAILABLE: 503,
    ErrorCode.INTERNAL: 500,
}


def default_status_for(code: ErrorCode) -> int:
    """HTTP status code for a given :class:`ErrorCode`."""
    return _CODE_TO_STATUS[code]


class APIError(Exception):
    """A typed error with a stable code, human message, and structured details.

    Endpoints raise ``APIError`` instead of ``HTTPException``. The global
    exception handler installed in :mod:`llm.serving.api` converts it to
    the standard envelope.

    Args:
        code: One of :class:`ErrorCode` (or a custom string for new codes).
        message: One-line human description.
        status_code: HTTP status to return. Defaults to the canonical
            status for ``code``; override only if you have a strong reason.
        details: Optional structured context (e.g. field-level validation
            errors, retry-after seconds, etc.).
    """

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        *,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if isinstance(code, ErrorCode):
            resolved_code = code.value
            resolved_status = status_code if status_code is not None else _CODE_TO_STATUS[code]
        else:
            resolved_code = str(code)
            resolved_status = status_code if status_code is not None else 500
        self.code = resolved_code
        self.message = message
        self.status_code = resolved_status
        self.details: dict[str, Any] = details or {}


def to_envelope(
    *,
    code: str,
    message: str,
    request_id: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical error envelope dict.

    Args:
        code: Machine-readable error code (e.g. ``"invalid_request"``).
        message: One-line human description.
        request_id: The X-Request-ID for this request.
        details: Optional structured context.

    Returns:
        ``{"error": {"code", "message", "details", "request_id"}}``
    """
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id,
        }
    }


def envelope_from_http_exception(exc: HTTPException, request_id: str) -> dict[str, Any]:
    """Convert a ``fastapi.HTTPException`` to the envelope shape.

    The detail field may be a string or a dict; we pass it through as
    ``details`` so callers don't lose context.
    """
    details: dict[str, Any] = {}
    if isinstance(exc.detail, dict):
        details = exc.detail
    elif exc.detail is not None:
        details = {"detail": exc.detail}
    # Map well-known statuses to a stable code. Unknown statuses map to
    # "internal" (so operators see something they can search for).
    status_to_code = {
        400: "invalid_request",
        401: "unauthorized",
        403: "unauthorized",
        404: "not_found",
        408: "timeout",
        422: "invalid_request",
        429: "rate_limited",
        500: "internal",
        502: "model_unavailable",
        503: "model_unavailable",
        504: "timeout",
    }
    code = status_to_code.get(exc.status_code, "internal")
    return to_envelope(
        code=code,
        message=str(exc.detail) if exc.detail is not None else code,
        request_id=request_id,
        details=details,
    )


def envelope_from_validation_error(exc: RequestValidationError, request_id: str) -> dict[str, Any]:
    """Convert a pydantic / FastAPI ``RequestValidationError`` to the envelope."""
    return to_envelope(
        code="invalid_request",
        message="Request validation failed.",
        request_id=request_id,
        details={"errors": exc.errors()},
    )


def envelope_from_api_error(exc: APIError, request_id: str) -> dict[str, Any]:
    """Convert an :class:`APIError` to the envelope shape."""
    return to_envelope(
        code=exc.code,
        message=exc.message,
        request_id=request_id,
        details=exc.details,
    )


def envelope_from_unexpected(exc: Exception, request_id: str, *, logger=None) -> dict[str, Any]:
    """Convert an unexpected ``Exception`` to the envelope shape.

    Logs the exception with the request_id so operators can correlate.
    The envelope does NOT leak internal details to clients.
    """
    if logger is not None:
        logger.exception("unexpected error during request %s", request_id, exc_info=exc)
    return to_envelope(
        code="internal",
        message="Internal server error.",
        request_id=request_id,
        details={"type": type(exc).__name__},
    )


def get_request_id(request: Request) -> str:
    """Return the request_id stored on the request state.

    Falls back to ``"unknown"`` if the middleware hasn't run (which would
    be a programming error in production — see
    :class:`llm.serving.middleware.RequestIDMiddleware`).
    """
    return getattr(request.state, "request_id", "unknown")


def envelope_response(payload: dict[str, Any], status_code: int, request_id: str) -> JSONResponse:
    """Build the canonical JSON response: envelope body + ``X-Request-ID`` header."""
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers={"X-Request-ID": request_id},
    )


def register_exception_handlers(app: FastAPI, *, logger: Any | None = None) -> None:
    """Wire the standard FastAPI exception handlers onto ``app``.

    All errors raised inside the serving API (typed :class:`APIError`,
    FastAPI :class:`HTTPException`, pydantic validation errors, and
    unexpected exceptions) flow through this single registration point
    and come out shaped as :func:`to_envelope`.

    Args:
        app: The :class:`fastapi.FastAPI` instance to register on.
        logger: Optional logger to receive unexpected exceptions. If
            ``None``, unexpected errors are still converted to a 500
            envelope but not logged here (the :class:`RequestIDMiddleware`
            access log line still records the failure).
    """
    from llm.serving.errors import (
        envelope_from_api_error,
        envelope_from_http_exception,
        envelope_from_unexpected,
        envelope_from_validation_error,
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> Response:
        return envelope_response(
            envelope_from_validation_error(exc, get_request_id(request)),
            status_code=422,
            request_id=get_request_id(request),
        )

    @app.exception_handler(APIError)
    async def _api_error_handler(request: Request, exc: APIError) -> Response:
        """Typed errors raised from endpoints / dependencies.

        Registered explicitly (not via the generic ``Exception`` handler)
        because FastAPI dispatches dependency exceptions through the
        same handler chain but the generic handler loses the typed
        status code when an exception is raised from a dependency.
        """
        return envelope_response(
            envelope_from_api_error(exc, get_request_id(request)),
            status_code=exc.status_code,
            request_id=get_request_id(request),
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> Response:
        request_id = get_request_id(request)
        if isinstance(exc, HTTPException):
            return envelope_response(
                envelope_from_http_exception(exc, request_id),
                status_code=exc.status_code,
                request_id=request_id,
            )
        return envelope_response(
            envelope_from_unexpected(exc, request_id, logger=logger),
            status_code=500,
            request_id=request_id,
        )
