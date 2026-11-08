"""CLI entry point for the serving API.

``reload=True`` is intentionally disabled because uvicorn's file-watcher
conflicts with ``from llm.serving.api import app`` (the watch import path
is incompatible with production use). For local development with auto-reload,
run uvicorn directly::

    uvicorn llm.serving.api:app --reload --host 127.0.0.1 --port 8000

The function refuses to start when the server would bind to a non-loopback
address without an ``api_key`` configured. ``host=0.0.0.0`` without auth
exposes the inference endpoint to the network; this guard makes that
mistake fail loudly at startup rather than silently at runtime.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import typer

from llm.serving.auth import assert_safe_bind
from llm.serving.config import ServingConfig

logger = logging.getLogger(__name__)


# ``llm serve`` is a Typer command so ``--help`` works like ``llm train``
# (the script used to point at the bare :func:`main`, which ignores
# argv — ``llm serve --help`` silently launched uvicorn and held a CUDA
# device). The flags default to ``None`` so the full ``ServingConfig`` /
# ``LLM_SERVING_*`` env resolution in :func:`main` still applies when the
# user does not override a specific value.
app = typer.Typer(pretty_exceptions_show_locals=False, invoke_without_command=True)


@app.callback()
def serve(
    host: str | None = typer.Option(None, "--host", help="Bind host (default: LLM_SERVING_HOST / ServingConfig.host)."),
    port: int | None = typer.Option(None, "--port", help="Bind port (default: LLM_SERVING_PORT / ServingConfig.port)."),
    config_path: str | None = typer.Option(None, "--config", help="Path to a serving YAML config."),
) -> None:
    """Start the llm serve OpenAI-compatible inference server."""
    if config_path is None:
        cfg = ServingConfig()
    else:
        # ``ServingConfig.from_yaml`` returns the DEFAULTS for a missing path
        # (a library-call convenience, pinned by its own test). That is wrong
        # for an explicit ``--config``: the server then starts happily on
        # 127.0.0.1:8000 with a default model while the operator believes their
        # YAML is in effect — no warning, no non-zero exit. Validate here rather
        # than changing the library method, whose "absent means all defaults"
        # behaviour other callers may rely on.
        resolved = Path(config_path)
        if not resolved.is_file():
            typer.echo(
                f"Error: serving config not found: {config_path}. Check the --config value.",
                err=True,
            )
            raise typer.Exit(code=1)
        cfg = ServingConfig.from_yaml(resolved)
    if host is not None:
        cfg.host = host
    if port is not None:
        cfg.port = port
    main(cfg)


def main(config: ServingConfig | None = None) -> None:
    """Entry point for the ``llm serve`` CLI.

    Args:
        config: Optional pre-built :class:`ServingConfig`. When ``None``
            (the common CLI case), a fresh instance is constructed from
            env vars. The :mod:`llm.serving.api` wrapper passes its
            module-level config so test code that monkey-patches
            ``api.config.host`` / ``api.config.api_key`` exercises the
            guard without rebuilding a fresh config.
    """
    if config is None:
        config = ServingConfig()
    # Shared fail-closed guard — see llm.serving.auth.assert_safe_bind.
    assert_safe_bind(config.host, config.api_key)

    # Hand the resolved config to the app BEFORE uvicorn imports it.
    #
    # ``uvicorn.run`` takes this module's app by import STRING, so it may
    # (re)import ``llm.serving.api`` in this process; without this call the
    # app rebuilt its own ``ServingConfig`` from env vars and everything the
    # CLI resolved — a ``--config`` YAML, ``--host``/``--port``, an
    # explicitly passed config — was silently dropped. The security-relevant
    # half: a YAML ``api_key`` never reached ``auth.get_api_key``, which
    # reads ``llm.serving.api.config``, so ``/generate`` and ``/metrics``
    # served anonymously. Verified live before the fix (200 with no auth
    # header while the YAML set a key). The bind address looked right the
    # whole time because uvicorn gets host/port as its own arguments, which
    # is exactly why this went unnoticed.
    from llm.serving import api as _api

    _api.apply_config(config)

    reload = os.environ.get("LLM_SERVING_RELOAD", "").lower() in ("1", "true", "yes")
    # Imported lazily so ``main`` is cheap to import (e.g. for `--help`).
    import uvicorn

    uvicorn.run("llm.serving.api:app", host=config.host, port=config.port, reload=reload)


if __name__ == "__main__":
    app()
