"""Register built-in runtime plugins via setuptools entry points."""

from __future__ import annotations

from llm.runtime.model_factory import MODEL_REGISTRY
from llm.runtime.plugins import load_entry_point_registry

_builtins_registered = False


def ensure_builtins_registered() -> None:
    """Idempotently discover and register model builders from entry points."""
    global _builtins_registered
    if _builtins_registered:
        return

    load_entry_point_registry("llm.models", MODEL_REGISTRY)
    _builtins_registered = True
