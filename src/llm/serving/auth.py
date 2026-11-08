"""Authentication for the serving API.

Currently a single shared API key compared in constant time
(``hmac.compare_digest``) to avoid timing leaks. Supports both
``X-API-Key: <key>`` and ``Authorization: Bearer <key>`` headers.

A future multi-tenant extension can replace the body of :func:`get_api_key`
without changing call sites, as long as the return contract (the key on
success, raising :class:`APIError` with code ``unauthorized`` on failure)
is preserved.
"""

from __future__ import annotations

import hmac

from fastapi import Security
from fastapi.security.api_key import APIKeyHeader

from llm.serving.errors import APIError, ErrorCode

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
authorization_header = APIKeyHeader(name="Authorization", auto_error=False)


def _extract_bearer_token(auth_header: str | None) -> str | None:
    """Extract the token from a ``Bearer`` authorization header."""
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


async def get_api_key(
    api_key_header_value: str | None = Security(api_key_header),
    auth_header: str | None = Security(authorization_header),
) -> str | None:
    """Verify the API key from ``X-API-Key`` or ``Authorization: Bearer``.

    Comparison uses ``hmac.compare_digest`` to avoid leaking key bytes via
    timing. If the module-level ``config.api_key`` is unset, auth is
    disabled and the function returns ``None`` (the public-host guard in
    :mod:`llm.serving.cli` blocks starting the server on a non-loopback
    interface without auth).
    """
    from llm.serving.api import config as _config

    if not _config.api_key:
        return None

    expected = _config.api_key
    # Check X-API-Key header first.
    if api_key_header_value is not None and hmac.compare_digest(api_key_header_value, expected):
        return api_key_header_value

    # Check Bearer token.
    bearer = _extract_bearer_token(auth_header)
    if bearer is not None and hmac.compare_digest(bearer, expected):
        return bearer

    raise APIError(ErrorCode.UNAUTHORIZED, "Could not validate credentials")


def is_loopback(host: str) -> bool:
    """Return True if ``host`` is a loopback address.

    Covers ``127.0.0.0/8`` and ``::1``. Anything else (``0.0.0.0``, ``*``,
    LAN IPs, public hostnames) is treated as non-loopback.
    """
    if host in ("127.0.0.1", "localhost", "::1"):
        return True
    return bool(host.startswith("127."))


#: Substrings that mark a value as an un-replaced template placeholder.
#:
#: ``configs/serve_pretrained.yaml`` ships
#: ``api_key: REPLACE_WITH_RANDOM_32_BYTE_HEX_AT_STARTUP`` so the YAML is
#: structurally valid and the public-host guard passes when the preset is
#: inspected. Nothing rejects that value at startup, so a user who launches the
#: preset as-is runs a server whose **only credential is a string printed in
#: this repository** — protected against nobody who can read the source.
#: Verified end-to-end before this guard existed: with the placeholder as the
#: configured key, a request carrying exactly that documented string returned
#: ``200`` while an unauthenticated one returned ``403``, i.e. the key
#: "worked" and was public.
_PLACEHOLDER_MARKERS = (
    "REPLACE_WITH",
    "REPLACE_ME",
    "CHANGEME",
    "CHANGE_ME",
    "YOUR_API_KEY",
    "<your",
    "placeholder",
)


def is_placeholder_secret(api_key: str | None) -> bool:
    """Return True when ``api_key`` looks like an un-replaced template value.

    Matching is case-insensitive and substring-based: placeholders are written
    in many shapes (``REPLACE_WITH_...``, ``changeme``, ``<your-key>``), and a
    real random hex secret cannot contain any of these words.
    """
    if not api_key:
        return False
    lowered = api_key.lower()
    return any(marker.lower() in lowered for marker in _PLACEHOLDER_MARKERS)


def assert_safe_bind(
    host: str,
    api_key: str | None,
    *,
    source: str = "ServingConfig.host",
) -> None:
    """Fail-closed guard: refuse to serve anonymously on a non-loopback bind.

    The single home for the check every entry point validates — the
    ``llm serve`` CLI (:func:`llm.serving.cli.main`), the FastAPI lifespan
    that the Docker image's direct ``uvicorn llm.serving.api:app`` launch
    runs, etc. Because all of them now read the SAME ``host`` the server
    actually binds (``LLM_SERVING_HOST`` / ``ServingConfig.host``), a
    Docker/uvicorn launch that skipped the CLI can no longer bind
    ``0.0.0.0`` with ``api_key=None`` and serve instruction-generating
    endpoints fully anonymously. ``host=0.0.0.0`` without
    auth fails at startup rather than silently at runtime.

    A **placeholder** key is refused on any bind, not just a public one: it
    reads as "auth is on" while the credential is published in the repo, so
    the guard's purpose (stopping anonymous exposure) is defeated even on
    loopback the moment the operator forwards a port.

    Args:
        host: The bind address the server will use.
        api_key: The configured ``ServingConfig.api_key`` (``None`` when
            auth is disabled).
        source: Name used in the error message for context.
    """
    if is_placeholder_secret(api_key):
        raise RuntimeError(
            f"Refusing to start: {source} holds the placeholder value "
            f"{api_key!r}, which is published in this repository and is therefore not a "
            f"secret. Generate a real key, e.g. "
            f"`export LLM_SERVING_API_KEY=$(openssl rand -hex 32)`."
        )
    if not is_loopback(host) and not api_key:
        raise RuntimeError(
            f"Refusing to start: {source}={host!r} binds to a "
            f"non-loopback address but api_key is not set. Anonymous access on a "
            f"public interface is unsafe. Either set LLM_SERVING_HOST to a loopback "
            f"address (127.0.0.1) or set LLM_SERVING_API_KEY."
        )
