"""Safe-pickle helpers: framework allowlist for ``torch.load(weights_only=True)``.

Security contract (RIL ISS-170 / ISS-211): ``torch.load(..., weights_only=True)``
refuses to execute arbitrary ``__reduce__`` code at load time. Model blobs and
pre-tokenized artifact files that legitimately embed framework ``nn.Module``
classes need those classes allowlisted first via
:func:`register_framework_safe_globals`. The allowlist covers every
``torch.nn.Module`` subclass defined in the framework packages plus the
``torch.nn.modules`` namespace, so real model artifacts load normally while a
pickle referencing ``os.system`` / ``subprocess`` / arbitrary builtins is
refused before any code runs.

Extracted from ``llm/serving/loader.py`` so the ``llm-quantize`` CLI (which
loads the same class of user-supplied blobs) can reuse the single allowlist
without pulling the FastAPI-heavy serving stack into the CLI process.
"""

from __future__ import annotations

import contextlib
import importlib
import pkgutil

import torch

_SAFE_GLOBALS_REGISTERED = False

#: Packages whose ``torch.nn.Module`` subclasses are trusted for reconstruction.
_FRAMEWORK_PACKAGES = ("llm.core", "llm.models", "llm.quantization")


def register_framework_safe_globals() -> None:
    """Allowlist every framework ``torch.nn.Module`` subclass (idempotent).

    Intentionally broad (every ``nn.Module`` subclass in those packages) so new
    attention/MLP/quantized-layer variants are covered without a hand-maintained
    list; the packages are small and already imported by the model loading path.
    Callers that legitimately load a *custom* non-framework class must register
    it themselves with ``torch.serialization.add_safe_globals`` before loading.
    """
    global _SAFE_GLOBALS_REGISTERED
    if _SAFE_GLOBALS_REGISTERED:
        return

    classes: dict[int, type] = {}

    def _collect_module_classes(pkg_name: str) -> None:
        """Collect every ``torch.nn.Module`` subclass defined in ``pkg_name``."""
        for mod in pkgutil.walk_packages(
            importlib.import_module(pkg_name).__path__,
            prefix=pkg_name + ".",
        ):
            with contextlib.suppress(Exception):
                # Optional deps (flash_attn etc.) may be absent — skip, don't
                # fail the caller over a missing optional dependency.
                module = importlib.import_module(mod.name)
                for obj in vars(module).values():
                    if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
                        classes[id(obj)] = obj

    # Framework built-ins (the classes a model blob / checkpoint may embed).
    with contextlib.suppress(Exception):
        for pkg_name in _FRAMEWORK_PACKAGES:
            _collect_module_classes(pkg_name)
    # torch.nn container/layer classes embedded by any nn.Module graph
    # (nn.Embedding, nn.Linear, nn.Dropout, nn.LayerNorm, nn.GELU, ...).
    # They have no code-execution surface under weights_only, so allowlisting
    # the whole built-in module namespace is safe.
    with contextlib.suppress(Exception):
        _collect_module_classes("torch.nn.modules")

    torch.serialization.add_safe_globals(list(classes.values()))
    _SAFE_GLOBALS_REGISTERED = True


__all__ = ["register_framework_safe_globals"]
