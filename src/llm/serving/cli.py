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

from llm.serving.auth import is_loopback
from llm.serving.config import ServingConfig

logger = logging.getLogger(__name__)


def main(config: ServingConfig | None = None) -> None:
    """Entry point for the ``llm-serve`` CLI.

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
    if not is_loopback(config.host) and not config.api_key:
        raise RuntimeError(
            f"Refusing to start: ServingConfig.host='{config.host}' binds to a "
            f"non-loopback address but api_key is not set. Anonymous access on a "
            f"public interface is unsafe. Either set LLM_SERVING_HOST to a loopback "
            f"address (127.0.0.1) or set LLM_SERVING_API_KEY."
        )

    reload = os.environ.get("LLM_SERVING_RELOAD", "").lower() in ("1", "true", "yes")
    # Imported lazily so ``main`` is cheap to import (e.g. for `--help`).
    import uvicorn

    uvicorn.run("llm.serving.api:app", host=config.host, port=8000, reload=reload)


if __name__ == "__main__":
    main()
