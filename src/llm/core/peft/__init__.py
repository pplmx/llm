"""Unified PEFT method registry (T2 PEFT #43).

This subpackage wraps the seven built-in PEFT methods
(``lora`` / ``qlora`` / ``adalora`` / ``prefix_tuning`` / ``ia3`` /
``bitfit`` / ``adapter``) behind a single registry, mirroring the
plugin-kernel pattern already used by :data:`llm.export.registry.EXPORT_REGISTRY`,
:data:`llm.generation.registry.BACKEND_REGISTRY`, and
:data:`llm.core.registry.ATTENTION_REGISTRY`.

The per-method helper modules (``llm.core.lora``, ``llm.core.ia3``,
``llm.core.bitfit``, ...) keep their module-level ``apply_*`` /
``merge_*`` / ``get_*_parameters`` / etc. functions unchanged — the
registry is an **additive** layer that holds a uniform
:class:`PEFTMethod` record for each built-in. New PEFT methods
(built-in or third-party via the ``llm.peft_methods`` setuptools
entry-point group) register by populating the same dataclass.

Public surface:
    :data:`PEFT_REGISTRY` — the underlying :class:`llm.runtime.registry.Registry`.
    :class:`PEFTMethod` — the dataclass every method must satisfy.
    :func:`apply_peft` — the canonical apply-time dispatch
        (``apply_peft(model, "lora", rank=8, alpha=16.0)``).
    :func:`merge_peft` / :func:`unmerge_peft` / :func:`disable_peft` /
    :func:`enable_peft` — inference-time helpers.
    :func:`get_peft_parameters` / :func:`count_peft_parameters` —
        introspection helpers.
    :func:`ensure_methods_registered` — idempotent bootstrap (loads
        ``llm.peft_methods`` entry points).
"""

from __future__ import annotations

from llm.core.peft.registry import (
    PEFT_REGISTRY,
    apply_peft,
    count_peft_parameters,
    disable_peft,
    enable_peft,
    ensure_methods_registered,
    get_peft_parameters,
    merge_peft,
    unmerge_peft,
)
from llm.core.peft.types import PEFTMethod, TargetModuleFilter

__all__ = [
    "PEFT_REGISTRY",
    "PEFTMethod",
    "TargetModuleFilter",
    "apply_peft",
    "count_peft_parameters",
    "disable_peft",
    "enable_peft",
    "ensure_methods_registered",
    "get_peft_parameters",
    "merge_peft",
    "unmerge_peft",
]
