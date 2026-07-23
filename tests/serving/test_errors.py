"""Unit tests for :mod:`llm.serving.errors`.

Covers the envelope helpers (to_envelope, envelope_from_*),
``get_request_id`` fallback, ``envelope_response``, ``APIError``
construction, ``default_status_for`` mapping, and the ``register_exception_handlers``
registration point. These had 63.5 % coverage; the gaps were the
envelope *functions* themselves, not the registration wiring.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from llm.serving.errors import (
    APIError,
    ErrorCode,
    default_status_for,
    envelope_from_api_error,
    envelope_from_http_exception,
    envelope_from_unexpected,
    envelope_from_validation_error,
    envelope_response,
    get_request_id,
    register_exception_handlers,
    to_envelope,
)

# --------------------------------------------------------------------------- #
# to_envelope
# --------------------------------------------------------------------------- #


class TestToEnvelope:
    def test_envelope_shape(self):
        env = to_envelope(
            code="invalid_request",
            message="bad input",
            request_id="req-1",
            details={"field": "x"},
        )
        assert env == {
            "error": {
                "code": "invalid_request",
                "message": "bad input",
                "details": {"field": "x"},
                "request_id": "req-1",
            }
        }

    def test_envelope_no_details_defaults_to_empty(self):
        env = to_envelope(
            code="internal",
            message="err",
            request_id="req-2",
        )
        assert env["error"]["details"] == {}

    def test_envelope_none_details_defaults_to_empty(self):
        env = to_envelope(
            code="internal",
            message="err",
            request_id="req-3",
            details=None,
        )
        assert env["error"]["details"] == {}


# --------------------------------------------------------------------------- #
# APIError construction
# --------------------------------------------------------------------------- #


class TestAPIError:
    def test_with_error_code_uses_status_map(self):
        err = APIError(ErrorCode.INVALID_REQUEST, "bad")
        assert err.code == "invalid_request"
        assert err.status_code == 400

    def test_with_custom_string_code_defaults_500(self):
        err = APIError("custom_code", "bad")
        assert err.code == "custom_code"
        assert err.status_code == 500

    def test_explicit_status_code_overrides_map(self):
        err = APIError(ErrorCode.INTERNAL, "bad", status_code=418)
        assert err.status_code == 418

    def test_explicit_status_code_for_custom_code(self):
        err = APIError("custom", "bad", status_code=422)
        assert err.status_code == 422

    def test_details_default_empty(self):
        err = APIError(ErrorCode.INTERNAL, "bad")
        assert err.details == {}

    def test_details_explicit(self):
        err = APIError(ErrorCode.INTERNAL, "bad", details={"k": "v"})
        assert err.details == {"k": "v"}

    def test_message_set(self):
        err = APIError(ErrorCode.INTERNAL, "custom message")
        assert err.message == "custom message"


# --------------------------------------------------------------------------- #
# default_status_for
# --------------------------------------------------------------------------- #


class TestDefaultStatusFor:
    @pytest.mark.parametrize(
        ("code", "status"),
        [
            (ErrorCode.INVALID_REQUEST, 400),
            (ErrorCode.UNAUTHORIZED, 403),
            (ErrorCode.TIMEOUT, 504),
            (ErrorCode.MODEL_UNAVAILABLE, 503),
            (ErrorCode.INTERNAL, 500),
        ],
    )
    def test_mapping(self, code: ErrorCode, status: int):
        assert default_status_for(code) == status


# --------------------------------------------------------------------------- #
# envelope_from_http_exception
# --------------------------------------------------------------------------- #


class TestEnvelopeFromHttpException:
    def test_string_detail(self):
        exc = HTTPException(status_code=400, detail="bad request")
        env = envelope_from_http_exception(exc, "req-1")
        assert env["error"]["code"] == "invalid_request"
        assert env["error"]["message"] == "bad request"
        assert env["error"]["details"] == {"detail": "bad request"}

    def test_dict_detail(self):
        exc = HTTPException(status_code=404, detail={"field": "missing"})
        env = envelope_from_http_exception(exc, "req-2")
        assert env["error"]["code"] == "not_found"
        assert env["error"]["details"] == {"field": "missing"}

    def test_401_unauthorized(self):
        exc = HTTPException(status_code=401, detail="unauth")
        env = envelope_from_http_exception(exc, "req-3")
        assert env["error"]["code"] == "unauthorized"

    def test_403_unauthorized(self):
        exc = HTTPException(status_code=403, detail="forbidden")
        env = envelope_from_http_exception(exc, "req-4")
        assert env["error"]["code"] == "unauthorized"

    def test_422_validation(self):
        exc = HTTPException(status_code=422, detail="val")
        env = envelope_from_http_exception(exc, "req-5")
        assert env["error"]["code"] == "invalid_request"

    def test_429_rate_limited(self):
        exc = HTTPException(status_code=429, detail="slow")
        env = envelope_from_http_exception(exc, "req-6")
        assert env["error"]["code"] == "rate_limited"

    def test_500_internal(self):
        exc = HTTPException(status_code=500, detail="err")
        env = envelope_from_http_exception(exc, "req-7")
        assert env["error"]["code"] == "internal"

    def test_502_model_unavailable(self):
        exc = HTTPException(status_code=502, detail="bad gw")
        env = envelope_from_http_exception(exc, "req-8")
        assert env["error"]["code"] == "model_unavailable"

    def test_503_model_unavailable(self):
        exc = HTTPException(status_code=503, detail="unavail")
        env = envelope_from_http_exception(exc, "req-9")
        assert env["error"]["code"] == "model_unavailable"

    def test_504_timeout(self):
        exc = HTTPException(status_code=504, detail="timeout")
        env = envelope_from_http_exception(exc, "req-10")
        assert env["error"]["code"] == "timeout"

    def test_unknown_status_maps_to_internal(self):
        exc = HTTPException(status_code=700, detail="odd")
        env = envelope_from_http_exception(exc, "req-11")
        assert env["error"]["code"] == "internal"
        assert env["error"]["message"] == "odd"

    def test_default_detail_is_status_text(self):
        # FastAPI/Starlette sets detail=None to the HTTP status text.
        exc = HTTPException(status_code=400, detail=None)
        env = envelope_from_http_exception(exc, "req-12")
        assert env["error"]["code"] == "invalid_request"
        assert env["error"]["details"] == {"detail": "Bad Request"}


# --------------------------------------------------------------------------- #
# envelope_from_validation_error
# --------------------------------------------------------------------------- #


class TestEnvelopeFromValidationError:
    def test_envelope_format(self):
        exc = RequestValidationError(errors=[{"type": "missing", "loc": ["x"]}])
        env = envelope_from_validation_error(exc, "req-1")
        assert env["error"]["code"] == "invalid_request"
        assert env["error"]["message"] == "Request validation failed."
        assert env["error"]["details"]["errors"] == [{"type": "missing", "loc": ["x"]}]
        assert env["error"]["request_id"] == "req-1"


# --------------------------------------------------------------------------- #
# envelope_from_api_error
# --------------------------------------------------------------------------- #


class TestEnvelopeFromApiError:
    def test_envelope_passes_through_fields(self):
        exc = APIError(
            code="custom",
            message="custom err",
            status_code=400,
            details={"layer": 1},
        )
        env = envelope_from_api_error(exc, "req-1")
        assert env["error"]["code"] == "custom"
        assert env["error"]["message"] == "custom err"
        assert env["error"]["details"] == {"layer": 1}
        assert env["error"]["request_id"] == "req-1"


# --------------------------------------------------------------------------- #
# envelope_from_unexpected
# --------------------------------------------------------------------------- #


class TestEnvelopeFromUnexpected:
    def test_no_logger(self):
        exc = ValueError("boom")
        env = envelope_from_unexpected(exc, "req-1")
        assert env["error"]["code"] == "internal"
        assert env["error"]["message"] == "Internal server error."
        assert env["error"]["details"]["type"] == "ValueError"

    def test_with_logger(self):
        exc = RuntimeError("crash")
        logger = logging.getLogger("test")
        env = envelope_from_unexpected(exc, "req-2", logger=logger)
        assert env["error"]["code"] == "internal"
        assert env["error"]["details"]["type"] == "RuntimeError"


# --------------------------------------------------------------------------- #
# get_request_id
# --------------------------------------------------------------------------- #


class TestGetRequestId:
    def test_returns_request_id_from_state(self):
        request = MagicMock()
        request.state.request_id = "req-123"
        assert get_request_id(request) == "req-123"

    def test_falls_back_to_unknown(self):
        request = MagicMock()
        # Simulate attribute not set (getattr default)
        request.state = MagicMock()
        # Remove request_id to trigger the getattr default
        del request.state.request_id
        assert get_request_id(request) == "unknown"


# --------------------------------------------------------------------------- #
# envelope_response
# --------------------------------------------------------------------------- #


class TestEnvelopeResponse:
    def test_builds_json_response_with_headers(self):
        payload = to_envelope(code="internal", message="err", request_id="req-1")
        resp = envelope_response(payload, status_code=500, request_id="req-1")
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 500
        assert resp.headers["X-Request-ID"] == "req-1"


# --------------------------------------------------------------------------- #
# register_exception_handlers
# --------------------------------------------------------------------------- #


class TestRegisterExceptionHandlers:
    def test_registers_handlers_without_error(self):
        app = FastAPI()
        register_exception_handlers(app)
        # If no exception is raised, the handlers were registered.
        # FastAPI stores handlers in app.exception_handlers
        assert len(app.exception_handlers) > 0

    def test_registers_with_logger(self):
        app = FastAPI()
        logger = logging.getLogger("test.register")
        register_exception_handlers(app, logger=logger)
        assert len(app.exception_handlers) > 0
