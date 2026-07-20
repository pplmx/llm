"""PEFT serving-side adapter loading (T2 PEFT #49).

Bridges the training-side PEFT save/load surface
(:mod:`llm.core.peft.checkpoint`) into the serving loader
(:mod:`llm.serving.loader`). Trained adapters (LoRA / IA³ / BitFit /
Adapter / Pfeiffer / AdaLoRA / QLoRA / Prefix Tuning) are loaded at
serve time and (optionally) folded into the base weights to save
runtime.

Two helpers are exposed:

- :func:`load_peft_into_model` — apply the method if needed and copy
  the saved adapter tensors into the model. Always safe to call
  (auto-applies when the destination model is fresh).
- :func:`merge_peft_into_model` — fold the adapter into the base
  weights for the methods that expose a merge helper (lora / adalora /
  ia3 / adapter / pfeiffer_adapter). For methods without a merge
  helper (bitfit / qlora / prefix_tuning) raises
  :class:`NotImplementedError` rather than silently no-op'ing — the
  caller must set ``peft_merge=False`` for those methods (and the
  config validator rejects ``peft_merge=True`` for them at startup,
  so this path only fires when the caller bypasses the validator).

Why two helpers instead of one ``load_and_merge`` flag: the caller
typically wants to load first, verify the adapter is in place, then
merge — separate calls surface partial-failure states more clearly
than a single boolean that conflates two steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm.core.peft import load_peft, merge_peft
from llm.core.peft.registry import ensure_methods_registered

if TYPE_CHECKING:
    import torch.nn as nn


def load_peft_into_model(
    model: nn.Module,
    method_name: str,
    adapter_path: str | Path,
    **override_kwargs: Any,
) -> nn.Module:
    """Apply ``method_name`` to ``model`` and load ``adapter_path``.

    Thin wrapper over :func:`llm.core.peft.load_peft` that exists so
    the serving loader has a single, named entry point to import (and
    so future serving-specific post-load steps — like eager
    device-move, dtype cast, or compilation — have a place to live
    without churning the call sites).

    Args:
        model: Destination model. If ``method_name`` is not yet
            applied, :func:`apply_peft` is called using the kwargs
            stored in the sidecar (overridable via
            ``override_kwargs``).
        method_name: Registered PEFT method name (e.g. ``"lora"``,
            ``"ia3"``, ``"bitfit"``).
        adapter_path: Path to a file written by
            :func:`llm.core.peft.save_peft`.
        **override_kwargs: Override individual kwargs from the
            sidecar (e.g. ``rank=16`` to widen a LoRA adapter when
            loading).

    Returns:
        The same ``model`` with adapter parameters populated
        (chainable).

    Raises:
        FileNotFoundError: If ``adapter_path`` doesn't exist.
        ValueError: If ``method_name`` is unknown or the sidecar's
            method-name / format-version metadata doesn't match.
        RuntimeError: If the model's adapter parameter count doesn't
            match the sidecar (architecture drift).
    """
    # Idempotent — load_peft handles the apply-if-needed step, but
    # calling ``ensure_methods_registered`` here makes the unknown-
    # method error message clean (no race with later bootstrap).
    ensure_methods_registered()
    return load_peft(model, adapter_path, method_name, **override_kwargs)


def merge_peft_into_model(
    model: nn.Module,
    method_name: str,
) -> nn.Module:
    """Fold ``method_name``'s adapter into the base weights.

    Useful at serve time when the adapter is unlikely to be
    disabled/ablated and folding saves the per-token cost of routing
    through the wrapper.

    Args:
        model: Model with ``method_name`` already applied.
        method_name: Registered PEFT method name.

    Returns:
        The same ``model`` with the adapter folded in (chainable).

    Raises:
        NotImplementedError: If ``method_name`` does not expose a
            merge helper (bitfit / qlora / prefix_tuning).
        ValueError: If ``method_name`` is unknown.
    """
    # ``merge_peft`` already raises NotImplementedError for methods
    # without a merge helper — the wrapper exists for symmetry with
    # :func:`load_peft_into_model` and so the call site in
    # ``load_model_and_tokenizer`` reads as a single coordinated pair.
    return merge_peft(model, method_name)


__all__ = [
    "load_peft_into_model",
    "merge_peft_into_model",
]
