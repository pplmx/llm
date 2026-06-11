"""Register built-in runtime plugins."""

from __future__ import annotations

from llm.runtime.model_factory import MODEL_REGISTRY, build_decoder
from llm.runtime.plugins import load_entry_point_registry

_builtins_registered = False


def ensure_builtins_registered() -> None:
    """Idempotently register built-in model builders and discover entry points."""
    global _builtins_registered
    if _builtins_registered:
        return

    if "decoder" not in MODEL_REGISTRY:
        MODEL_REGISTRY.register("decoder", build_decoder)

    load_entry_point_registry("llm.models", MODEL_REGISTRY)
    _builtins_registered = True
