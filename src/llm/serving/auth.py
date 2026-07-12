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
