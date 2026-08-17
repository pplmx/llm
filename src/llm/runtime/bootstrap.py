"""Register built-in runtime plugins via setuptools entry points."""

from __future__ import annotations

import threading

from llm.runtime.model_factory import MODEL_REGISTRY
from llm.runtime.plugins import load_entry_point_registry

_builtins_registered = False
_registration_lock = threading.Lock()


def ensure_builtins_registered() -> None:
    """Idempotently discover and register model builders from entry points."""
    global _builtins_registered
    if _builtins_registered:
        return

    # Double-checked locking (RIL ISS-119 pattern, see generation/registry.py):
    # the guard above is the hot path; the lock serializes the cold-start race
    # (two threads both passing the bare flag, both loading entry points, the
    # second thread's Registry.register("decoder", ...) raising ValueError), so
    # a concurrent caller re-checks the flag inside the critical section.
    with _registration_lock:
        if _builtins_registered:
            return

        load_entry_point_registry("llm.models", MODEL_REGISTRY)
        _builtins_registered = True
