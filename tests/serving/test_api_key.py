"""Unit tests for ``get_api_key`` without booting the FastAPI lifespan.

Bypasses ``TestClient`` so the engine is never loaded — pure-function tests of the
auth dependency. Regression coverage for the ``hmac.compare_digest`` switch
(Finding AS) and the T2 #16 module split (auth dependency now lives in
``llm.serving.auth``).
"""

from __future__ import annotations

import asyncio

import pytest

from llm.serving import api
from llm.serving.errors import APIError, ErrorCode


@pytest.fixture
def set_api_key():
    """Set a known ``config.api_key`` and restore on teardown."""
    original = api.config.api_key
    api.config.api_key = "secret-key-123"
    yield "secret-key-123"
    api.config.api_key = original


def _run(coro):
    return asyncio.run(coro)


def test_get_api_key_returns_none_when_unconfigured():
    """If no key is configured, auth is a no-op (returns None)."""
    original = api.config.api_key
    api.config.api_key = None
    try:
        assert _run(api.get_api_key(api_key_header_value=None, auth_header=None)) is None
    finally:
        api.config.api_key = original


def test_get_api_key_accepts_correct_x_api_key_header(set_api_key):
    assert _run(api.get_api_key(api_key_header_value=set_api_key, auth_header=None)) == set_api_key


def test_get_api_key_accepts_correct_bearer_token(set_api_key):
    assert _run(api.get_api_key(api_key_header_value=None, auth_header=f"Bearer {set_api_key}")) == set_api_key


def test_get_api_key_rejects_wrong_x_api_key(set_api_key):  # noqa: ARG001
    with pytest.raises(APIError) as exc_info:
        _run(api.get_api_key(api_key_header_value="wrong-key", auth_header=None))
    assert exc_info.value.status_code == 403
    assert exc_info.value.code == ErrorCode.UNAUTHORIZED.value


def test_get_api_key_rejects_wrong_bearer_token(set_api_key):  # noqa: ARG001
    with pytest.raises(APIError) as exc_info:
        _run(api.get_api_key(api_key_header_value=None, auth_header="Bearer wrong-token"))
    assert exc_info.value.status_code == 403
    assert exc_info.value.code == ErrorCode.UNAUTHORIZED.value


def test_get_api_key_rejects_missing_headers_when_key_required(set_api_key):  # noqa: ARG001
    """No headers + key configured -> 403, never crash."""
    with pytest.raises(APIError) as exc_info:
        _run(api.get_api_key(api_key_header_value=None, auth_header=None))
    assert exc_info.value.status_code == 403


def test_get_api_key_uses_constant_time_compare(set_api_key):
    """Verify behavior is identical to ``hmac.compare_digest`` semantics: a partial
    prefix that equals the start of the configured key still fails.

    This is a smoke test, not a timing test (those are flaky in CI).
    """
    prefix = set_api_key[:5]
    with pytest.raises(APIError):
        _run(api.get_api_key(api_key_header_value=prefix, auth_header=None))


# --- main() guard: refuse non-loopback without api_key ---


class TestIsLoopback:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.42"])
    def test_loopback_hosts(self, host):
        assert api.is_loopback(host) is True

    @pytest.mark.parametrize("host", ["0.0.0.0", "", "192.168.1.10", "10.0.0.1", "example.com"])  # noqa: S104
    def test_non_loopback_hosts(self, host):
        assert api.is_loopback(host) is False


def test_main_refuses_non_loopback_without_api_key():
    """``main()`` must refuse to start when host is non-loopback and no api_key."""
    from llm.serving import api

    # Force non-loopback host, clear api_key
    original_host = api.config.host
    original_key = api.config.api_key
    api.config.host = "0.0.0.0"  # noqa: S104
    api.config.api_key = None
    try:
        with pytest.raises(RuntimeError, match="Refusing to start"):
            api.main()
    finally:
        api.config.host = original_host
        api.config.api_key = original_key


def test_main_allows_non_loopback_with_api_key(monkeypatch):
    """When api_key is configured, non-loopback bind is allowed."""
    from llm.serving import api

    started: dict = {}

    def fake_run(*args, **kwargs):
        started.update(kwargs)
        started["args"] = args

    monkeypatch.setattr("uvicorn.run", fake_run, raising=False)

    original_host = api.config.host
    original_key = api.config.api_key
    api.config.host = "0.0.0.0"  # noqa: S104
    api.config.api_key = "some-secret"
    try:
        api.main()
    finally:
        api.config.host = original_host
        api.config.api_key = original_key

    assert started.get("host") == "0.0.0.0"  # noqa: S104


def test_main_allows_loopback_without_api_key(monkeypatch):
    """Loopback bind is allowed without an api_key (dev/local use case)."""
    from llm.serving import api

    started: dict = {}

    def fake_run(*args, **kwargs):
        started.update(kwargs)
        started["args"] = args

    monkeypatch.setattr("uvicorn.run", fake_run, raising=False)

    original_host = api.config.host
    original_key = api.config.api_key
    api.config.host = "127.0.0.1"
    api.config.api_key = None
    try:
        api.main()
    finally:
        api.config.host = original_host
        api.config.api_key = original_key

    assert started.get("host") == "127.0.0.1"
