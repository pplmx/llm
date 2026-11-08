"""``llm serve --config`` must not silently fall back to defaults.

``ServingConfig.from_yaml`` deliberately returns the default config for a
missing path (an "absent config means all defaults" convenience used by library
callers, pinned by ``test_missing_file_returns_defaults``). That is the wrong
answer for an EXPLICIT ``--config``: the server started on the defaults —
default model, default host/port — while the operator believed their YAML was
in effect. No warning, no non-zero exit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from llm.serving.cli import serve


@pytest.fixture
def command():
    """Wrap the bare ``serve`` function in a one-command Typer app."""
    app = typer.Typer()
    app.command()(serve)
    return CliRunner(), app


def test_missing_config_path_exits_nonzero_with_an_actionable_message(command, tmp_path: Path):
    runner, app = command
    missing = tmp_path / "does_not_exist.yaml"

    result = runner.invoke(app, ["--config", str(missing)])

    assert result.exit_code != 0, "an explicit --config pointing nowhere must not start the server"
    output = result.output
    assert "not found" in output.lower(), f"the error should say the config is missing, got: {output!r}"
    assert str(missing) in output, "the error should echo the path the user typed"


def test_directory_as_config_path_is_rejected(command, tmp_path: Path):
    """A directory exists, so ``exists()`` alone would let it through."""
    runner, app = command
    a_directory = tmp_path / "config_dir"
    a_directory.mkdir()

    result = runner.invoke(app, ["--config", str(a_directory)])

    assert result.exit_code != 0, "a directory is not a usable config file"


def test_valid_config_file_is_loaded(command, tmp_path: Path, monkeypatch):
    """The guard must not reject a real file; the contents must reach the config."""
    runner, app = command
    config_file = tmp_path / "serve.yaml"
    config_file.write_text("host: 127.0.0.1\nport: 8123\nmodel_path: null\n")

    captured: list = []

    def _fake_main(config) -> None:
        captured.append(config)

    monkeypatch.setattr("llm.serving.cli.main", _fake_main)

    result = runner.invoke(app, ["--config", str(config_file)])

    assert result.exit_code == 0, f"a valid config must be accepted, got: {result.output!r}"
    assert captured, "main() was never called"
    assert captured[0].port == 8123, "the YAML value did not reach the config"


# --------------------------------------------------------------------------- #
# The loaded config must actually reach the app uvicorn serves
# --------------------------------------------------------------------------- #


class TestLoadedConfigReachesTheApp:
    """A ``--config`` file is worthless if the served app never sees it.

    ``uvicorn.run`` takes the app by import **string**, so the CLI's
    ``ServingConfig`` and the app's module-level ``config`` are separate objects
    unless something connects them. Nothing did: ``apply_config`` did not
    exist, and ``test_valid_config_file_is_loaded`` above monkeypatches
    ``main`` to a stub — so it proved the YAML reached ``main()`` and could say
    nothing about what the running server then used.

    The consequence was security-relevant rather than cosmetic. Measured live
    with a YAML setting ``api_key: secret-in-yaml``: ``/generate`` and
    ``/metrics`` both answered **200 to a request with no auth header**, because
    ``auth.get_api_key`` reads ``llm.serving.api.config`` (``api_key=None`` from
    the discarded default) while the YAML value died in the CLI's local object.
    The bind address looked correct throughout, since uvicorn receives host/port
    as its own arguments — which is precisely why the loss went unnoticed.
    """

    def test_apply_config_replaces_the_modules_config(self):
        """``apply_config`` must swap the object every consumer reads."""
        from llm.serving import api
        from llm.serving.config import ServingConfig

        original = api.config
        replacement = ServingConfig(api_key="from-yaml", port=8123, max_concurrent_requests=1)
        try:
            api.apply_config(replacement)
            assert api.config is replacement
            assert api.config.api_key == "from-yaml"
            # ``auth.get_api_key`` reads this exact attribute path.
            from llm.serving.api import config as auth_sees

            assert auth_sees.api_key == "from-yaml"
        finally:
            api.apply_config(original)

    def test_apply_config_rebuilds_the_concurrency_semaphore(self):
        """The semaphore's bound is frozen at construction — it must be rebuilt.

        A ``max_concurrent_requests`` of 4 left in place while the YAML says 1
        means the configured cap is silently 4x too generous; mutating the
        attribute could not fix it because ``asyncio.Semaphore`` fixes its
        value at construction.
        """
        from llm.serving import api
        from llm.serving.config import ServingConfig

        original = api.config
        try:
            api.apply_config(ServingConfig(max_concurrent_requests=1))
            assert api.inference_semaphore._value == 1
            api.apply_config(ServingConfig(max_concurrent_requests=7))
            assert api.inference_semaphore._value == 7
        finally:
            api.apply_config(original)

    def test_app_requires_the_key_after_apply_config(self, monkeypatch):
        """End-to-end: after ``apply_config``, an unauthenticated request is 403.

        This is the assertion whose absence let the bug live: it goes through
        the real ASGI app and the real auth dependency instead of inspecting
        config objects, so it fails if *any* link between the CLI's config and
        the serving path is dropped.
        """
        from fastapi.testclient import TestClient

        from llm.serving import api
        from llm.serving.batch_engine import ContinuousBatchingEngine
        from llm.serving.config import ServingConfig
        from llm.serving.generation_service import ServingGenerationService

        original = api.config
        monkeypatch.setattr(
            ServingGenerationService,
            "from_config",
            classmethod(lambda cls, config, **kw: _fake_service()),
        )
        monkeypatch.setattr(
            ContinuousBatchingEngine,
            "from_serving_config",
            classmethod(lambda cls, config, **kw: _fake_engine()),
        )
        monkeypatch.setattr(api, "_log_server_config", lambda *a, **kw: None)
        try:
            api.apply_config(ServingConfig(api_key="from-yaml"))
            with TestClient(api.app) as client:
                # No auth header at all.
                assert client.post("/generate", json={"prompt": "hi"}).status_code == 403
                # The YAML key works.
                assert (
                    client.post("/generate", json={"prompt": "hi"}, headers={"X-API-Key": "from-yaml"}).status_code
                    == 200
                )
        finally:
            api.apply_config(original)

    def test_body_size_cap_follows_the_applied_config(self, monkeypatch):
        """``max_request_bytes`` must track ``apply_config``, not import time.

        The middleware is built at import, so an ``int`` captured there freezes
        whatever the environment held before the CLI handed its config over.
        """
        from fastapi.testclient import TestClient

        from llm.serving import api
        from llm.serving.batch_engine import ContinuousBatchingEngine
        from llm.serving.config import ServingConfig
        from llm.serving.generation_service import ServingGenerationService

        original = api.config
        monkeypatch.setattr(
            ServingGenerationService,
            "from_config",
            classmethod(lambda cls, config, **kw: _fake_service()),
        )
        monkeypatch.setattr(
            ContinuousBatchingEngine,
            "from_serving_config",
            classmethod(lambda cls, config, **kw: _fake_engine()),
        )
        monkeypatch.setattr(api, "_log_server_config", lambda *a, **kw: None)
        try:
            api.apply_config(ServingConfig(max_request_bytes=256))
            with TestClient(api.app, raise_server_exceptions=False) as client:
                oversized = client.post("/generate", content=b"{}", headers={"content-length": "4096"})
                assert oversized.status_code == 413
                # A body the DEFAULT cap would have allowed is fine here too —
                # the point is the tight cap is in force, not that it shrank.
                under = client.post("/generate", json={"prompt": "hi"}, headers={"content-length": "16"})
                assert under.status_code != 413
        finally:
            api.apply_config(original)


def _fake_service():
    from unittest.mock import MagicMock

    service = MagicMock()
    service.tokenizer.encode = lambda text: [1]
    service.tokenizer.decode = lambda ids: "x"
    service.generate.return_value = "ok"
    service.stream.return_value = iter([])
    return service


def _fake_engine():
    from unittest.mock import MagicMock

    return MagicMock()


def test_main_hands_its_config_to_the_app(monkeypatch):
    """``main()`` itself must call ``apply_config`` — not just the function work alone.

    Every test above calls ``apply_config`` directly, so they all pass even if
    ``main`` never calls it (verified by deleting the call: the suite stayed
    green). That is the same shape as the bug being fixed — the pieces work,
    the wire between them is what breaks — so the wiring needs its own test.

    ``uvicorn`` is stubbed out: this asserts the hand-off, and leaving the real
    server out keeps the test offline and instant. Host/port go to
    ``uvicorn.run`` directly, which is why the bind address looked correct even
    while the rest of the config was being dropped.
    """
    import sys
    import types

    from llm.serving import api, cli
    from llm.serving.config import ServingConfig

    captured: dict = {}

    fake_uvicorn = types.ModuleType("uvicorn")

    def _run(app_path, host=None, port=None, **_kwargs):
        captured["app_path"] = app_path
        captured["host"] = host
        captured["port"] = port
        # What the app would read at this point, i.e. what the hand-off achieved.
        captured["app_config"] = api.config

    fake_uvicorn.run = _run
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    original = api.config
    try:
        cli.main(ServingConfig(host="127.0.0.1", port=8123, api_key="from-yaml"))
        assert captured["app_path"] == "llm.serving.api:app"
        # The app's module-level config IS the one main() resolved.
        assert captured["app_config"].api_key == "from-yaml"
        assert captured["app_config"].port == 8123
    finally:
        api.apply_config(original)
