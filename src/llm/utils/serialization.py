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

import importlib
import logging
import pkgutil

import torch

logger = logging.getLogger(__name__)

_SAFE_GLOBALS_REGISTERED = False

#: Packages whose ``torch.nn.Module`` subclasses are trusted for reconstruction.
_FRAMEWORK_PACKAGES = ("llm.core", "llm.models", "llm.quantization", "llm.multimodal")


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
    # Submodules that failed to import (expected for soft/optional deps like
    # flash_attn). Each one degrades the allowlist, so it is logged loud and
    # registration is left UNCOMPLETED — a later call retries instead of
    # permanently pinning a partial allowlist (RIL ISS-243).
    failures: list[str] = []

    def _collect_module_classes(pkg_name: str) -> None:
        """Collect every ``torch.nn.Module`` subclass defined in ``pkg_name``."""
        try:
            paths = importlib.import_module(pkg_name).__path__
        except Exception as exc:  # noqa: BLE001 — a broken pkg is loud, not fatal
            failures.append(f"{pkg_name}: {exc!r}")
            return
        for mod in pkgutil.walk_packages(paths, prefix=pkg_name + "."):
            try:
                module = importlib.import_module(mod.name)
            except Exception as exc:  # noqa: BLE001
                # Optional deps (flash_attn etc.) may be absent — skip, but
                # record it so the caller knows the allowlist is partial.
                failures.append(f"{mod.name}: {exc!r}")
                continue
            for obj in vars(module).values():
                if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
                    classes[id(obj)] = obj

    # Framework built-ins (the classes a model blob / checkpoint may embed).
    for pkg_name in _FRAMEWORK_PACKAGES:
        _collect_module_classes(pkg_name)
    # torch.nn container/layer classes embedded by any nn.Module graph
    # (nn.Embedding, nn.Linear, nn.Dropout, nn.LayerNorm, nn.GELU, ...).
    # They have no code-execution surface under weights_only, so allowlisting
    # the whole built-in module namespace is safe.
    _collect_module_classes("torch.nn.modules")

    if classes or not failures:
        # Register whatever was found (idempotent — overlapping repeats are
        # merged by torch); a real add_safe_globals failure PROPAGATES, never
        # swallowed. The flag is pinned only on a clean walk so a partial
        # allowlist self-heals on the next call.
        torch.serialization.add_safe_globals(list(classes.values()))
    if failures:
        logger.warning(
            "safe-globals allowlist is PARTIAL: %d submodule(s) failed to import "
            "(soft deps absent, or a broken module). Registration is NOT finalised "
            "so the next call retries. First failures: %s",
            len(failures),
            "; ".join(failures[:5]),
        )
    else:
        _SAFE_GLOBALS_REGISTERED = True


__all__ = ["register_framework_safe_globals"]
