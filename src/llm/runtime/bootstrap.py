"""Register built-in runtime plugins."""

from __future__ import annotations

from llm.runtime.model_factory import MODEL_REGISTRY, build_decoder

_builtins_registered = False


def ensure_builtins_registered() -> None:
    """Idempotently register built-in model builders."""
    global _builtins_registered
    if _builtins_registered:
        return

    if "decoder" not in MODEL_REGISTRY:
        MODEL_REGISTRY.register("decoder", build_decoder)

    _builtins_registered = True
