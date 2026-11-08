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
    _CODE_TO_STATUS,
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
            (ErrorCode.QUEUE_FULL, 503),
            (ErrorCode.MODEL_UNAVAILABLE, 503),
            (ErrorCode.PAYLOAD_TOO_LARGE, 413),
            (ErrorCode.INTERNAL, 500),
        ],
    )
    def test_mapping(self, code: ErrorCode, status: int):
        assert default_status_for(code) == status

    def test_every_code_has_a_status(self):
        """Every ``ErrorCode`` member must be mapped, and only to a valid status.

        The parametrized list above can silently fall behind when a new member
        is added: ``QUEUE_FULL`` and ``PAYLOAD_TOO_LARGE`` were both absent from
        it, so a code could ship with no mapping and nothing failed. A missing
        key surfaces as a ``KeyError`` inside error *handling* — turning the
        response a client can act on into an opaque 500.
        """
        for member in ErrorCode:
            status = default_status_for(member)
            assert 400 <= status <= 599, f"{member} maps to a non-error status {status}"
        assert set(ErrorCode) == set(_CODE_TO_STATUS), "an ErrorCode member has no HTTP status mapping"


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

    def test_405_is_method_not_allowed_not_internal(self):
        """A wrong-method request is a CLIENT fault, not ``internal``.

        405 was absent from the status mapping, so every ``GET`` on a
        POST-only route reported ``code: "internal"`` — telling an integrator
        that a request aimed at the wrong verb (or the wrong URL prefix) was a
        server-side failure and sending them to look in the wrong place. The
        code is the documented branch point.
        """
        exc = HTTPException(status_code=405, detail="Method Not Allowed")
        env = envelope_from_http_exception(exc, "req-405")
        assert env["error"]["code"] == "method_not_allowed"

    def test_413_is_payload_too_large(self):
        """A framework-raised 413 must share the middleware's code."""
        exc = HTTPException(status_code=413, detail="too big")
        env = envelope_from_http_exception(exc, "req-413")
        assert env["error"]["code"] == ErrorCode.PAYLOAD_TOO_LARGE.value

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

    def test_ctx_carrying_a_valueerror_is_serializable(self):
        """A ``value_error`` entry's ``ctx`` holds the EXCEPTION OBJECT.

        pydantic keeps the original exception so callers can re-raise it. Fed
        straight into ``JSONResponse`` that raises
        ``TypeError: Object of type ValueError is not JSON serializable``
        **inside the 422 handler**, which surfaced as a 500 "internal error"
        for every request that tripped a custom validator (`stop` longer than
        4 strings, unsupported chat knobs, an empty prompt) — the client-caused
        422 was reported as a server fault and the actionable message was
        lost.
        """
        import json

        raw_error = {
            "type": "value_error",
            "loc": ("body", "stop"),
            "msg": "Value error, stop list may contain at most 4 strings",
            "input": ["a", "a", "a", "a", "a"],
            "ctx": {"error": ValueError("stop list may contain at most 4 strings")},
        }
        exc = RequestValidationError(errors=[raw_error])

        env = envelope_from_validation_error(exc, "req-1")

        # The whole envelope must be JSON-encodable — that is what
        # ``JSONResponse`` does with it.
        json.dumps(env)
        rendered = env["error"]["details"]["errors"][0]
        assert rendered["msg"] == "Value error, stop list may contain at most 4 strings"
        assert rendered["ctx"]["error"]["type"] == "ValueError"
        assert rendered["ctx"]["error"]["message"] == "stop list may contain at most 4 strings"

    def test_ctx_and_input_are_recursively_rendered(self):
        """Nested non-primitive values are coerced, not left to blow up later.

        A validator can put anything in ``ctx``; the envelope must stay total
        because a validation error about arbitrary client input can never
        itself be allowed to fail.
        """
        import json

        raw_error = {
            "type": "value_error",
            "loc": ("body",),
            "msg": "bad",
            "input": {"nested": {"deep": (1, 2)}},
            "ctx": {"error": KeyError("missing-key"), "extra": {"set": {1}}},
        }
        env = envelope_from_validation_error(RequestValidationError(errors=[raw_error]), "req-2")

        json.dumps(env)
        rendered = env["error"]["details"]["errors"][0]
        assert rendered["ctx"]["error"] == {"type": "KeyError", "message": "'missing-key'"}
        assert rendered["input"] == {"nested": {"deep": [1, 2]}}
        # A set is rendered to a list (order is irrelevant for one element).
        assert rendered["ctx"]["extra"]["set"] == [1]

    def test_json_safe_primitives_pass_through_unchanged(self):
        """Already-JSON-safe error entries must not be mangled."""
        errors = [
            {"type": "missing", "loc": ["body", "prompt"], "msg": "Field required", "input": None},
            {"type": "too_short", "loc": ["body", "messages"], "msg": "at least 1", "ctx": {"min_length": 1}},
        ]
        env = envelope_from_validation_error(RequestValidationError(errors=errors), "req-3")

        assert env["error"]["details"]["errors"] == [
            {"type": "missing", "loc": ["body", "prompt"], "msg": "Field required", "input": None},
            {"type": "too_short", "loc": ["body", "messages"], "msg": "at least 1", "ctx": {"min_length": 1}},
        ]


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

    def test_does_not_leak_exception_class_name(self):
        """Regression: the envelope must NOT carry
        ``type(exc).__name__`` — it let a client fingerprint the framework and
        backend class from any 500."""
        exc = RuntimeError("crash")
        env = envelope_from_unexpected(exc, "req-3")
        assert env["error"]["code"] == "internal"
        assert "type" not in str(env["error"].get("details", {}))
        assert "RuntimeError" not in str(env)

    def test_with_logger(self):
        exc = RuntimeError("crash")
        logger = logging.getLogger("test")
        env = envelope_from_unexpected(exc, "req-2", logger=logger)
        assert env["error"]["code"] == "internal"


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

    def test_registers_a_handler_for_starlette_http_exception(self):
        """Framework 404/405 must resolve to THIS handler, not Starlette's.

        Starlette installs a default handler for ``StarletteHTTPException``;
        FastAPI's ``HTTPException`` subclasses it. A handler registered only
        for the subclass never runs, because dispatch resolves by MRO and the
        base-class handler wins. The unit tests for
        ``envelope_from_http_exception`` all passed while that function was
        never called on a real request — so the registration itself has to be
        asserted.
        """
        from starlette.exceptions import HTTPException as StarletteHTTPException

        app = FastAPI()
        register_exception_handlers(app)
        assert StarletteHTTPException in app.exception_handlers

    def test_real_404_and_405_use_the_envelope(self):
        """End-to-end: the framework's own errors must be enveloped.

        404 (unknown path) and 405 (wrong method) are the failures a client
        integrating against this API hits first, yet they were the only
        responses answering ``{"detail": ...}`` instead of the documented
        ``{"error": {code, message, details, request_id}}``.
        """
        from fastapi.testclient import TestClient

        app = FastAPI()
        register_exception_handlers(app)

        @app.post("/only-post")
        async def _only_post() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(app) as client:
            not_found = client.get("/does-not-exist")
            assert not_found.status_code == 404
            assert not_found.json()["error"]["code"] == "not_found"
            assert not_found.json()["error"]["request_id"] is not None

            wrong_method = client.get("/only-post")
            assert wrong_method.status_code == 405
            assert wrong_method.json()["error"]["code"] == "method_not_allowed"
            # The Allow header is what makes 405 actionable.
            assert wrong_method.headers["Allow"] == "POST"

    def test_real_405_preserves_the_allow_header(self):
        """Wrapping a 405 must not drop ``Allow``.

        ``Allow`` is the entire actionable content of a 405 — it tells the
        client which verb the route takes. An envelope wrapper that built a
        fresh response lost it, leaving a 405 the client could not act on.
        """
        from fastapi.testclient import TestClient

        app = FastAPI()
        register_exception_handlers(app)

        @app.post("/only-post")
        async def _only_post() -> dict[str, bool]:
            return {"ok": True}

        with TestClient(app) as client:
            response = client.put("/only-post")
            assert response.status_code == 405
            assert response.headers.get("Allow") == "POST"
            assert response.headers["X-Request-ID"]


# --------------------------------------------------------------------------- #
# Body request_id vs the X-Request-ID header
# --------------------------------------------------------------------------- #


class TestRequestIdConsistency:
    """The body's ``request_id`` must always equal the ``X-Request-ID`` header.

    ``docs/guides/inference.md`` states it plainly: "``request_id`` echoes the
    ``X-Request-ID`` response header, so a client-reported failure can be found
    in the server's access log". That promise is what makes a client-reported
    id useful — an operator greps the access log for the id the client quoted.

    Nothing tested it. Each error path builds its response in a different place
    (the typed handler, the framework handler, the validation handler, and the
    body-size middleware, which never reaches an exception handler at all), so
    they can drift apart one at a time and every existing test would stay
    green — the per-path tests each check only their own half.
    """

    @staticmethod
    def _app():
        from fastapi import FastAPI

        from llm.serving.errors import APIError, ErrorCode
        from llm.serving.middleware import RequestBodySizeLimit, RequestIDMiddleware

        app = FastAPI()
        # Mirrors the production wiring in ``llm.serving.api``: the size cap is
        # added first (inner), the request-id middleware second (outer).
        app.add_middleware(RequestBodySizeLimit, max_bytes=1024)
        app.add_middleware(RequestIDMiddleware)
        register_exception_handlers(app)

        @app.get("/api-error")
        async def _api_error() -> dict[str, bool]:
            raise APIError(ErrorCode.INVALID_REQUEST, "bad")

        @app.post("/echo")
        async def _echo(payload: dict) -> dict[str, bool]:
            # The body is validated (not used): the test only needs the route
            # to parse a JSON body so a malformed one yields a real 422.
            del payload
            return {"ok": True}

        return app

    @pytest.mark.parametrize(
        ("label", "send"),
        [
            ("api_error", lambda c: c.get("/api-error")),
            ("validation", lambda c: c.post("/echo", content=b"{not json")),
            ("framework_404", lambda c: c.get("/nope")),
            ("framework_405", lambda c: c.get("/echo")),
            (
                "payload_too_large",
                lambda c: c.post("/echo", content=b"{}", headers={"content-length": "99999"}),
            ),
        ],
    )
    def test_body_request_id_matches_the_header(self, label, send):
        from fastapi.testclient import TestClient

        with TestClient(self._app(), raise_server_exceptions=False) as client:
            response = send(client)

        body_id = response.json()["error"]["request_id"]
        header_id = response.headers.get("X-Request-ID")
        assert header_id, f"{label}: no X-Request-ID header"
        assert body_id == header_id, f"{label}: body request_id {body_id!r} != header {header_id!r}"

    def test_an_inbound_request_id_is_echoed_in_both_places(self):
        """A client-supplied id must come back in the body AND the header.

        Clients stitch retries together by sending their own id, so the id they
        get back has to be theirs — not a freshly generated one — on both
        surfaces.
        """
        from fastapi.testclient import TestClient

        inbound = "abcdef0123456789abcdef0123456789"
        with TestClient(self._app(), raise_server_exceptions=False) as client:
            response = client.get("/nope", headers={"X-Request-ID": inbound})

        assert response.headers["X-Request-ID"] == inbound
        assert response.json()["error"]["request_id"] == inbound


# --------------------------------------------------------------------------- #
# Retry guidance headers
# --------------------------------------------------------------------------- #


class TestRetryGuidanceHeaders:
    """Transient failures must tell the client WHEN to retry, not just that it may.

    OpenAI's SDK checks ``x-should-retry`` first and ``Retry-After`` second,
    both **before** its status-code heuristic — and that heuristic retries
    *every* 5xx. So a bare 503 already gets retried, but on the SDK's own curve
    (``INITIAL_RETRY_DELAY`` 0.5s, capped at 8s), which is far too eager against
    a queue that needs a full batching step to drain. Measured against the SDK's
    real ``_should_retry``: a bare 503 parses ``Retry-After`` as ``None``; the
    response this server now sends parses it as ``1.0``.

    The header also matters in the other direction: a 503 whose cause is
    permanent (a dead model) should not be retried at all, and ``x-should-retry:
    false`` is the only way to say that, because the status code cannot.
    """

    def test_api_error_emits_retry_after_rounded_up(self):
        """A fractional delay rounds UP — never advertise a shorter wait."""
        from llm.serving.errors import APIError, ErrorCode

        err = APIError(ErrorCode.QUEUE_FULL, "full", retry_after=2.3)
        assert err.response_headers()["Retry-After"] == "3"

    def test_api_error_never_emits_zero_retry_after(self):
        """``Retry-After: 0`` would invite an immediate retry into a full queue."""
        from llm.serving.errors import APIError, ErrorCode

        err = APIError(ErrorCode.QUEUE_FULL, "full", retry_after=0.0)
        assert err.response_headers()["Retry-After"] == "1"

    def test_api_error_emits_should_retry_both_ways(self):
        from llm.serving.errors import APIError, ErrorCode

        assert APIError(ErrorCode.QUEUE_FULL, "x", should_retry=True).response_headers() == {"x-should-retry": "true"}
        assert APIError(ErrorCode.INTERNAL, "x", should_retry=False).response_headers() == {"x-should-retry": "false"}

    def test_api_error_without_guidance_emits_no_headers(self):
        """Codes that opt out must not grow headers by accident."""
        from llm.serving.errors import APIError, ErrorCode

        assert APIError(ErrorCode.INVALID_REQUEST, "bad").response_headers() == {}

    def test_the_sdk_honours_the_headers_we_send(self):
        """Assert against the SDK's policy, not our own reading of the spec.

        The SDK is the consumer; if its ``_should_retry`` did not parse
        ``Retry-After``, sending the header would be decorative. Constructed
        without a client object because the SDK's bundled httpx shim cannot
        build a base URL in this environment — the policy methods only touch
        the response and module constants.
        """
        httpx = pytest.importorskip("httpx2")
        import openai._base_client as base_client
        from openai._base_client import BaseClient

        class _Probe:
            _should_retry = BaseClient._should_retry
            _parse_retry_after_header = BaseClient._parse_retry_after_header
            MAX_RETRY_AFTER_DELAY = base_client.MAX_RETRY_AFTER_DELAY

        probe = _Probe()

        def ask(status: int, headers: dict[str, str]) -> tuple[bool, float | None]:
            response = httpx.Response(status, headers=headers, request=httpx.Request("POST", "http://x/"))
            return probe._should_retry(response), probe._parse_retry_after_header(response.headers)

        # What the server now sends for queue_full.
        retried, delay = ask(503, {"retry-after": "1", "x-should-retry": "true"})
        assert retried is True
        assert delay == 1.0, "the SDK must actually read our Retry-After"

        # What it used to send (bare 503): retried, but with no server-chosen delay.
        assert ask(503, {}) == (True, None)

        # An explicit "do not retry" wins over the 5xx heuristic.
        assert ask(500, {"x-should-retry": "false"})[0] is False


class TestQueueFullCarriesRetryGuidance:
    """The live response for a full queue must carry both headers."""

    @staticmethod
    def _client(monkeypatch):
        from fastapi.testclient import TestClient

        from llm.serving import api
        from llm.serving.batch_engine import ContinuousBatchingEngine
        from llm.serving.generation_service import ServingGenerationService
        from llm.serving.scheduler import Scheduler

        fake_service = MagicMock()
        fake_service.tokenizer.encode = lambda text: [1]
        fake_service.generate.return_value = "ok"
        # A FULL scheduler is what the route's pre-check inspects.
        engine = MagicMock()
        engine.scheduler = MagicMock(spec=Scheduler)
        engine.scheduler.is_full.return_value = True
        engine.scheduler.max_waiting = 4
        fake_service.backend.engine = engine
        fake_service.backend.engine = engine

        monkeypatch.setattr(
            ServingGenerationService,
            "from_config",
            classmethod(lambda cls, config, **kw: fake_service),
        )
        monkeypatch.setattr(
            ContinuousBatchingEngine,
            "from_serving_config",
            classmethod(lambda cls, config, **kw: MagicMock()),
        )
        monkeypatch.setattr(api, "_log_server_config", lambda *a, **kw: None)

        client = TestClient(api.app, raise_server_exceptions=False)
        monkeypatch.setattr("llm.serving.routers.generate.generation_service", fake_service)
        return client

    @pytest.mark.parametrize(
        ("path", "body"),
        [
            ("/generate", {"prompt": "hi", "max_new_tokens": 2}),
            ("/v1/chat/completions", {"model": "llm", "messages": [{"role": "user", "content": "hi"}]}),
            ("/v1/completions", {"model": "llm", "prompt": "hi", "max_tokens": 2}),
        ],
    )
    def test_queue_full_response_has_retry_headers(self, monkeypatch, path, body):
        """All three generation entry points, not just ``/generate``."""
        client = self._client(monkeypatch)
        with client:
            response = client.post(path, json=body)

        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "queue_full"
        assert response.headers.get("Retry-After") == "1"
        assert response.headers.get("x-should-retry") == "true"


# --------------------------------------------------------------------------- #
# KV-slot exhaustion is backpressure, not a broken model
# --------------------------------------------------------------------------- #


class TestNoFreeSlotsIsRetryableBackpressure:
    """Two distinct capacity limits exist; only one used to be a type.

    ``QueueFullError`` bounds requests *waiting to be admitted*; KV-slot
    exhaustion bounds those *holding cache memory*. Both mean "at capacity,
    come back shortly", but the slot case raised a bare ``RuntimeError``, which
    fell into the routers' generic catch-all and was reported as
    ``model_unavailable`` — "the model is broken, give up" — with no retry
    guidance. A client following that advice abandons a request that would have
    succeeded moments later.

    ``NoFreeSlotsError`` makes the two conditions distinguishable by type, the
    same way ``QueueFullError`` already did for the waiting queue.
    """

    def test_allocator_raises_the_typed_error(self):
        from llm.serving.batch_engine import NoFreeSlotsError, SlotAllocator

        allocator = SlotAllocator(total_slots=1)
        allocator.allocate("first")
        with pytest.raises(NoFreeSlotsError):
            allocator.allocate("second")

    def test_the_type_is_a_runtime_error_subclass(self):
        """Existing ``except RuntimeError`` handlers must keep catching it.

        The typed catch has to be placed BEFORE the generic one in each router;
        keeping the inheritance means a route that forgets still degrades to the
        old behaviour rather than letting the exception escape as a 500.
        """
        from llm.serving.batch_engine import NoFreeSlotsError

        assert issubclass(NoFreeSlotsError, RuntimeError)

    @pytest.mark.parametrize(
        ("path", "body", "generator"),
        [
            ("/generate", {"prompt": "hi", "max_new_tokens": 2}, "generate"),
            (
                "/v1/chat/completions",
                {"model": "llm", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
                "generate",
            ),
            ("/v1/completions", {"model": "llm", "prompt": "hi", "max_tokens": 2}, "generate"),
        ],
    )
    def test_every_generation_route_maps_it_to_retryable_queue_full(self, monkeypatch, path, body, generator):
        """End-to-end: 503 + ``queue_full`` + retry headers, on all three routes."""
        from fastapi.testclient import TestClient

        from llm.serving import api
        from llm.serving.batch_engine import ContinuousBatchingEngine, NoFreeSlotsError
        from llm.serving.generation_service import ServingGenerationService

        fake_service = MagicMock()
        fake_service.tokenizer.encode = lambda text: [1]
        fake_service.generate.side_effect = NoFreeSlotsError("all 2 KV cache slots are in use")
        fake_service.batch_generate.side_effect = NoFreeSlotsError("all 2 KV cache slots are in use")
        fake_service.stream.side_effect = NoFreeSlotsError("all 2 KV cache slots are in use")

        monkeypatch.setattr(
            ServingGenerationService,
            "from_config",
            classmethod(lambda cls, config, **kw: fake_service),
        )
        monkeypatch.setattr(
            ContinuousBatchingEngine,
            "from_serving_config",
            classmethod(lambda cls, config, **kw: MagicMock()),
        )
        monkeypatch.setattr(api, "_log_server_config", lambda *a, **kw: None)

        with TestClient(api.app, raise_server_exceptions=False) as client:
            monkeypatch.setattr("llm.serving.routers.generate.generation_service", fake_service)
            response = client.post(path, json=body)

        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "queue_full", (
            f"{path} reported {response.json()['error']['code']!r} — KV-slot exhaustion "
            "must not be reported as a model failure"
        )
        assert response.headers.get("Retry-After") == "1"
        assert response.headers.get("x-should-retry") == "true"

    def test_a_real_model_failure_still_maps_to_model_unavailable(self, monkeypatch):
        """The control: a genuine backend ``RuntimeError`` keeps its old mapping.

        Without this, the tests above would pass even if EVERY ``RuntimeError``
        had been reclassified as backpressure — which would tell clients to
        retry a permanently broken model forever.
        """
        from fastapi.testclient import TestClient

        from llm.serving import api
        from llm.serving.batch_engine import ContinuousBatchingEngine
        from llm.serving.generation_service import ServingGenerationService

        fake_service = MagicMock()
        fake_service.tokenizer.encode = lambda text: [1]
        fake_service.generate.side_effect = RuntimeError("CUDA device-side assert")

        monkeypatch.setattr(
            ServingGenerationService,
            "from_config",
            classmethod(lambda cls, config, **kw: fake_service),
        )
        monkeypatch.setattr(
            ContinuousBatchingEngine,
            "from_serving_config",
            classmethod(lambda cls, config, **kw: MagicMock()),
        )
        monkeypatch.setattr(api, "_log_server_config", lambda *a, **kw: None)

        with TestClient(api.app, raise_server_exceptions=False) as client:
            monkeypatch.setattr("llm.serving.routers.generate.generation_service", fake_service)
            response = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 2})

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
        # And it must NOT carry retry guidance it cannot honour.
        assert "Retry-After" not in response.headers


class TestNoFreeSlotsReachesTheRouterThroughTheRealEngine:
    """The exception must survive the engine's own machinery, not just a mock.

    The router-level tests above inject ``NoFreeSlotsError`` straight from a
    mocked generation service, which proves the router's ``except`` clause maps
    the type correctly — but not that the type ever ARRIVES. In production it is
    raised deep inside ``_lock_step_pre`` (during ``allocate``), travels out of
    ``step()``, and then out of ``stream_request``'s generator before the router
    sees anything. A ``finally`` that swallowed it, or a generator wrapper that
    converted it, would leave the router tests green and the live API reporting
    a generic 500.

    This drives a REAL engine with a deliberately exhausted allocator, so the
    raise happens on the real path.
    """

    @staticmethod
    def _exhausted_engine(device):
        import torch

        from llm.models.decoder import DecoderModel
        from llm.serving.batch_engine import ContinuousBatchingEngine
        from llm.serving.schemas import GenerationRequest
        from tests.support.tokenizers import StubTokenizer

        torch.manual_seed(0)
        model = DecoderModel(vocab_size=100, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=32, device=device)
        engine = ContinuousBatchingEngine(model=model, tokenizer=StubTokenizer(), max_batch_size=2, max_seq_len=32)
        request = GenerationRequest(prompt="hi", max_new_tokens=3)
        engine.add_request(request)
        # Take every slot so the next allocate() inside step() raises.
        while engine.slot_allocator.num_free > 0:
            engine.slot_allocator.allocate(f"occupied-{engine.slot_allocator.num_free}")
        return engine, request

    def test_step_raises_the_typed_error(self, device):
        from llm.serving.batch_engine import NoFreeSlotsError

        engine, _request = self._exhausted_engine(device)
        with pytest.raises(NoFreeSlotsError):
            engine.step()

    def test_stream_request_propagates_it_to_the_caller(self, device):
        """``stream_request`` must not swallow or rewrap it.

        That generator owns a ``finally`` which reaps sequences and releases
        slots on every exit path — the natural place for an exception to be
        lost. The router can only map the type if it reaches the caller intact.
        """
        from llm.serving.batch_engine import NoFreeSlotsError

        engine, request = self._exhausted_engine(device)
        with pytest.raises(NoFreeSlotsError):
            list(engine.stream_request(request))
