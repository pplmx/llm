"""Smoke tests: every public module in ``llm.serving`` imports cleanly.

Regression coverage for the T2 #16 module split (Finding H). If any of
the new modules has a circular import, missing dependency, or syntax
error, this test surfaces it before the router-level tests even get to
run.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "llm.serving.api",
    "llm.serving.auth",
    "llm.serving.batch_engine",
    "llm.serving.chat_template",
    "llm.serving.cli",
    "llm.serving.config",
    "llm.serving.errors",
    "llm.serving.generation_service",
    "llm.serving.loader",
    "llm.serving.middleware",
    "llm.serving.scheduler",
    "llm.serving.schemas",
    "llm.serving.routers",
    "llm.serving.routers.chat",
    "llm.serving.routers.generate",
    "llm.serving.routers.health",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    """Each listed module imports without raising."""
    importlib.import_module(module_name)


def test_app_builds_without_warnings() -> None:
    """The FastAPI app can be constructed from the split module graph."""
    from llm.serving.api import app

    assert app.title == "LLM Inference API"
    # Routers wired (one per logical group). Enumerate registered paths via the
    # generated OpenAPI schema rather than ``app.routes``: FastAPI >= 0.115
    # stores included routers as lazy ``_IncludedRouter`` wrappers whose
    # ``path`` attribute does not expose the underlying ``APIRoute`` paths, so
    # iterating ``app.routes`` would miss ``/health`` etc. even though they
    # are correctly wired. ``app.openapi()["paths"]`` is the stable,
    # version-independent view of registered routes.
    paths = set(app.openapi()["paths"])
    assert "/health" in paths
    assert "/generate" in paths
    assert "/v1/chat/completions" in paths
