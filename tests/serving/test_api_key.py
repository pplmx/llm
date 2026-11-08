"""Unit tests for ``get_api_key`` without booting the FastAPI lifespan.

Bypasses ``TestClient`` so the engine is never loaded — pure-function tests of the
auth dependency. Regression coverage for the ``hmac.compare_digest`` switch
(Finding AS) and the T2 #16 module split (auth dependency now lives in
``llm.serving.auth``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

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


# --- the shared fail-closed bind guard ---
#
# The Docker image runs ``uvicorn llm.serving.api:app --host $LLM_SERVING_HOST``
# directly — it never calls ``main()`` — so the guard had to move into the
# shared :func:`llm.serving.auth.assert_safe_bind` that BOTH ``cli.main`` and
# the FastAPI lifespan call, and the Docker bind had to be driven by the same
# ``LLM_SERVING_HOST`` value the guard validates. These tests pin that contract.


@pytest.mark.parametrize(
    ("host", "key"),
    [
        ("0.0.0.0", None),  # noqa: S104
        ("192.168.1.10", None),
        ("example.com", None),
    ],
)
def test_assert_safe_bind_refuses_public_bind_without_key(host, key):
    from llm.serving.auth import assert_safe_bind

    with pytest.raises(RuntimeError, match="Refusing to start"):
        assert_safe_bind(host, key)


@pytest.mark.parametrize(
    ("host", "key"),
    [
        ("127.0.0.1", None),
        ("localhost", None),
        ("::1", None),
        ("0.0.0.0", "some-secret"),  # noqa: S104
        ("192.168.1.10", "some-secret"),
    ],
)
def test_assert_safe_bind_allows_safe_combinations(host, key):
    from llm.serving.auth import assert_safe_bind

    assert_safe_bind(host, key)  # must not raise


def test_lifespan_calls_shared_guard_with_actual_bind(monkeypatch):
    """The lifespan must hand assert_safe_bind the exact config host+key, so a
    direct-uvicorn launch (the Docker path, which never calls main()) is
    guarded with the SAME policy as the CLI."""

    calls: list[tuple[str, str | None]] = []

    def fake_guard(host, key):
        calls.append((host, key))
        raise RuntimeError("Refusing to start (guard invoked)")

    # The lifespan imports the guard in-function, so patch the auth-module
    # attribute (the name both the lifespan and cli.main resolve at call time).
    import llm.serving.auth as auth_mod

    monkeypatch.setattr(auth_mod, "assert_safe_bind", fake_guard)

    original_host = api.config.host
    original_key = api.config.api_key
    api.config.host = "0.0.0.0"  # noqa: S104
    api.config.api_key = None
    try:
        import asyncio

        async def _enter() -> None:
            async with api.app.router.lifespan_context(api.app):
                pass

        with pytest.raises(RuntimeError, match="Refusing to start"):
            asyncio.run(_enter())
    finally:
        api.config.host = original_host
        api.config.api_key = original_key

    assert calls == [("0.0.0.0", None)]  # noqa: S104


# --- Dockerfile single-source-of-truth regression ---


def test_dockerfile_binds_from_llm_serving_host():
    """The Docker CMD must bind ``$LLM_SERVING_HOST`` (not hardcode 0.0.0.0) so
    the lifespan guard validates the SAME address uvicorn binds."""
    from pathlib import Path

    dockerfile = Path(__file__).resolve().parents[2] / "Dockerfile"
    assert dockerfile.exists(), "repo Dockerfile not found"
    text = dockerfile.read_text(encoding="utf-8")
    # The CMD must shell-expand the env var, not hardcode a literal bind.
    assert '--host "$LLM_SERVING_HOST"' in text or "--host $LLM_SERVING_HOST" in text
    # The default must be a public bind so a key-less `docker run` fails closed.
    assert 'ENV LLM_SERVING_HOST="0.0.0.0"' in text


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


def test_lifespan_refuses_non_loopback_without_api_key():
    """The *lifespan* (the uvicorn/Docker entrypoint) must refuse to start
    when host is non-loopback and no api_key — not just ``cli.main``.

    Regression: the Docker CMD runs ``uvicorn llm.serving.api:app`` directly,
    bypassing the CLI guard, so a container started without
    LLM_SERVING_API_KEY served /generate fully anonymously on 0.0.0.0. The
    guard moved into the app's lifespan so every entrypoint is covered.
    """
    from fastapi.testclient import TestClient

    original_host = api.config.host
    original_key = api.config.api_key
    api.config.host = "0.0.0.0"  # noqa: S104
    api.config.api_key = None
    try:
        # TestClient runs the lifespan on startup; the guard raises before
        # any real model load. We patch the model-load factory so the only
        # failure that can occur is the guard itself.
        import llm.serving.api as api_mod

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            api_mod.ServingGenerationService, "from_config", classmethod(lambda cls, config, **kw: None)
        )
        monkeypatch.setattr(
            api_mod.ContinuousBatchingEngine, "from_serving_config", classmethod(lambda cls, config, **kw: None)
        )
        try:
            with pytest.raises(RuntimeError, match="Refusing to start"), TestClient(api.app):
                pass
        finally:
            monkeypatch.undo()
    finally:
        api.config.host = original_host
        api.config.api_key = original_key


def test_lifespan_allows_loopback_without_api_key():
    """Loopback without api_key is the dev default and must still start.

    We only assert the startup guard passes; the rest of the lifespan
    (model load) is patched out so this stays a unit test of the guard.
    """
    from fastapi.testclient import TestClient

    original_host = api.config.host
    original_key = api.config.api_key
    api.config.host = "127.0.0.1"
    api.config.api_key = None
    try:
        import llm.serving.api as api_mod

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            api_mod.ServingGenerationService, "from_config", classmethod(lambda cls, config, **kw: None)
        )
        monkeypatch.setattr(
            api_mod.ContinuousBatchingEngine, "from_serving_config", classmethod(lambda cls, config, **kw: None)
        )
        monkeypatch.setattr(api_mod, "_log_server_config", lambda *a, **kw: None)
        try:
            # The loopback check must pass (no "Refusing to start"); the
            # lifespan then proceeds to model wiring, which is patched out to
            # None here, so startup fails later with an AttributeError from
            # ``generation_service.model``. Asserting that specific error
            # proves the guard did not fire first (the guard raises a
            # RuntimeError mentioning "Refusing to start", not this).
            with pytest.raises((AttributeError, TypeError), match="model"), TestClient(api.app):
                pass
        finally:
            monkeypatch.undo()
    finally:
        api.config.host = original_host
        api.config.api_key = original_key


# --- compose.yml must not ship a container that cannot start ---


def test_compose_passes_the_api_key_the_image_requires():
    """``just compose-up`` must not produce a container that dies at startup.

    The image defaults ``LLM_SERVING_HOST=0.0.0.0`` and the lifespan refuses to
    start without auth, so a compose file that omits the key builds a service
    that fails a few seconds after ``docker compose up`` — with the guard's
    error as the only explanation. Verified before this was fixed: reproducing
    the image's exact environment (host 0.0.0.0, no key) printed
    "Refusing to start: ... api_key is not set" and uvicorn exited.

    The ``:?`` form is asserted specifically: it turns the failure into a
    ``docker compose`` parse-time abort that NAMES the variable, instead of a
    container that starts and collapses later.
    """
    from pathlib import Path

    compose = Path(__file__).resolve().parents[2] / "compose.yml"
    assert compose.exists(), "repo compose.yml not found"
    env = yaml.safe_load(compose.read_text(encoding="utf-8"))["services"]["llm"]["environment"]

    # S104 suppressed: this ASSERTS the container's intentional public bind
    # string; it does not bind anything.
    assert env["LLM_SERVING_HOST"] == "0.0.0.0", "the image's fail-closed default must be made explicit"  # noqa: S104
    key_expr = str(env["LLM_SERVING_API_KEY"])
    # ``${VAR:?message}`` — required, with an actionable message.
    assert key_expr.startswith("${LLM_SERVING_API_KEY:?"), (
        "the API key must be REQUIRED via ${LLM_SERVING_API_KEY:?...}; "
        f"got {key_expr!r}. Without the ':?' a missing variable expands to empty "
        "and the container fails at runtime instead of at compose parse time."
    )
    # The message must say what to do, not just that something is missing.
    assert "LLM_SERVING_API_KEY" in key_expr


def test_compose_model_path_defaults_to_unset_not_a_broken_path():
    """An unset model path must expand to empty, which the loader treats as unset.

    ``load_model_and_tokenizer`` guards with ``if not config.model_path``, so an
    empty string is safe. A literal path default would instead make
    ``docker compose up`` try to load a checkpoint that does not exist.
    """
    from pathlib import Path

    compose = Path(__file__).resolve().parents[2] / "compose.yml"
    env = yaml.safe_load(compose.read_text(encoding="utf-8"))["services"]["llm"]["environment"]
    assert str(env["LLM_SERVING_MODEL_PATH"]).endswith(":-}"), (
        "LLM_SERVING_MODEL_PATH must default to empty (${VAR:-}); "
        "a concrete default would point at a checkpoint the image does not have"
    )


# --- A published placeholder must never act as a credential ---------------


class TestPlaceholderKeysAreRefused:
    """A placeholder in a shipped config is a published credential.

    ``configs/serve_pretrained.yaml`` used to ship
    ``api_key: REPLACE_WITH_RANDOM_32_BYTE_HEX_AT_STARTUP`` so the YAML would
    be structurally valid and the public-host guard would pass on inspection.
    Nothing rejected that value at startup, so launching the preset as-is ran a
    server whose ONLY credential was a string printed in this repository.
    Measured end-to-end before the guard existed: with that placeholder
    configured, a request carrying exactly the documented string returned
    ``200`` while an unauthenticated one returned ``403`` — auth was "on" and
    defeated simultaneously.

    The guard refuses a placeholder on ANY bind, not just a public one: an
    operator running loopback-only today with a port forward tomorrow would
    otherwise expose the endpoint with a known key.
    """

    @pytest.mark.parametrize(
        "placeholder",
        [
            "REPLACE_WITH_RANDOM_32_BYTE_HEX_AT_STARTUP",
            "replace_with_your_key",
            "changeme",
            "<your-api-key>",
            "YOUR_API_KEY",
            "placeholder-secret",
        ],
    )
    def test_placeholder_shapes_are_detected(self, placeholder):
        from llm.serving.auth import is_placeholder_secret

        assert is_placeholder_secret(placeholder) is True

    @pytest.mark.parametrize(
        "real_key",
        [
            None,
            "",
            # A real `openssl rand -hex 32`.
            "9f2c47a1b83d05e6c4712f8a0d3b96e5a7c21d4f60b8e39a5c7d21f08b3a6e4c",
        ],
    )
    def test_real_or_absent_keys_are_not_flagged(self, real_key):
        from llm.serving.auth import is_placeholder_secret

        assert is_placeholder_secret(real_key) is False

    @pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])  # noqa: S104
    def test_placeholder_refused_on_any_bind(self, host):
        """Loopback is NOT enough — see the class docstring."""
        from llm.serving.auth import assert_safe_bind

        with pytest.raises(RuntimeError, match="placeholder"):
            assert_safe_bind(host, "REPLACE_WITH_RANDOM_32_BYTE_HEX_AT_STARTUP")

    def test_the_shipped_preset_carries_no_placeholder_key(self):
        """The regression that prompted the guard: the file itself.

        The preset binds ``0.0.0.0``, so the startup guard already refuses it
        without a key — but only after this fix. Before it, the file's
        placeholder satisfied the "key is set" condition and the server came up
        with a public credential.
        """
        import yaml

        compose_dir = Path(__file__).resolve().parents[2] / "configs"
        raw = yaml.safe_load((compose_dir / "serve_pretrained.yaml").read_text(encoding="utf-8"))
        assert "api_key" not in raw, (
            "serve_pretrained.yaml must not set api_key at all: an explicit YAML "
            "value SHADOWS LLM_SERVING_API_KEY (pydantic-settings precedence), so even "
            "`api_key: null` would break the documented env-var workflow"
        )

    def test_env_var_supplies_the_key_when_yaml_omits_it(self, monkeypatch):
        """The documented workflow must actually work.

        Omitting the key from the YAML is what lets ``LLM_SERVING_API_KEY`` be
        the source. Verified from the real file, not a constructed config.
        """
        from llm.serving.config import ServingConfig

        monkeypatch.setenv("LLM_SERVING_API_KEY", "9f2c47a1b83d05e6c4712f8a0d3b96e5a7c21d4f60b8e39a5c7d21f08b3a6e4c")
        cfg = ServingConfig.from_yaml(Path(__file__).resolve().parents[2] / "configs" / "serve_pretrained.yaml")
        assert cfg.api_key == "9f2c47a1b83d05e6c4712f8a0d3b96e5a7c21d4f60b8e39a5c7d21f08b3a6e4c"
