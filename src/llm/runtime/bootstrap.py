"""Register built-in runtime plugins via setuptools entry points."""

from __future__ import annotations

import threading

from llm.runtime import model_factory
from llm.runtime.model_factory import MODEL_REGISTRY
from llm.runtime.plugins import load_entry_point_registry

_builtins_registered = False
_registration_lock = threading.Lock()


def ensure_builtins_registered() -> None:
    """Idempotently discover and register model builders from entry points.

    Entry-point discovery can resolve to an EMPTY registry — source-tree use
    without ``pip install -e .``, an outdated venv, or a subprocess with a
    different ``sys.path``. The two reference builtins live in this package,
    so they are registered as code-level fallbacks when discovery yielded
    nothing; only a total failure (no entry points AND no fallback) is an
    error, raised with the entry-point group named so the fix is obvious
    (RIL ISS-418).
    """
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
        if "decoder" not in MODEL_REGISTRY.names():
            # Reference built-ins always register (they are code in this
            # package, independent of entry-point metadata).
            MODEL_REGISTRY.register("decoder", model_factory.build_decoder)
            MODEL_REGISTRY.register("regression_mlp", model_factory.build_regression_mlp)
        if not MODEL_REGISTRY.names():
            raise RuntimeError(
                "llm.models entry points resolved to an EMPTY registry and the code-level "
                "fallbacks could not be registered. Install the package (`pip install -e .`) "
                "or fix sys.path so the 'llm.models' entry-point group resolves."
            )
        _builtins_registered = True
