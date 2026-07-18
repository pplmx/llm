"""PEFT method registry and dispatch (T2 PEFT #43).

Mirrors :mod:`llm.export.registry` so third-party PEFT methods can
plug in via the ``llm.peft_methods`` setuptools entry-point group
without forking :mod:`llm.core.peft`.

Built-in methods:
    ``lora``, ``qlora``, ``adalora``, ``prefix_tuning``, ``ia3``,
    ``bitfit``, ``adapter`` — registered eagerly by
    :func:`ensure_methods_registered`.

Usage:
    >>> from llm.core.peft import apply_peft, count_peft_parameters
    >>> apply_peft(model, "lora", rank=8, alpha=16.0)
    >>> trainable, total = count_peft_parameters(model, "lora")

Plugin authors register a method via ``pyproject.toml``::

    [project.entry-points."llm.peft_methods"]
    my_method = "my_pkg.peft:build_my_peft_method"

The factory ``build_my_peft_method()`` must return a
:class:`llm.core.peft.PEFTMethod` instance. Built-ins are registered
**before** the entry-point load — a plugin claiming a built-in name is
silently skipped (matches the ``EXPORT_REGISTRY`` convention;
``overwrite=True`` is reserved for explicit override paths).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from llm.core.peft.methods import iter_builtin_methods
from llm.core.peft.types import PEFTMethod
from llm.runtime.plugins import load_entry_point_registry
from llm.runtime.registry import Registry

if TYPE_CHECKING:
    import torch.nn as nn


PEFT_REGISTRY: Registry[PEFTMethod] = Registry("PEFTMethod")

# Idempotency guard — :func:`ensure_methods_registered` returns
# immediately on the second call so the bootstrap is cheap.
_methods_registered: bool = False


def ensure_methods_registered() -> None:
    """Idempotently register built-in methods and load entry points.

    Built-ins are registered **before** the entry-point load so a
    plugin that claims a built-in name raises / is silently skipped —
    the built-in is the source of truth. This matches the convention
    in :func:`llm.export.registry.ensure_exporters_registered` and
    :func:`llm.generation.registry.ensure_backends_registered`.
    """
    global _methods_registered
    if _methods_registered:
        return

    for method in iter_builtin_methods():
        # ``Registry.register`` raises on duplicate names. Built-ins
        # use stable names so re-import won't trigger duplicates;
        # third parties load after this point and the entry-point
        # loader defaults to ``overwrite=False`` so plugins claiming
        # built-in names are silently skipped (matching the export
        # convention).
        if method.name not in PEFT_REGISTRY:
            PEFT_REGISTRY.register(method.name, method)

    load_entry_point_registry("llm.peft_methods", PEFT_REGISTRY)
    _methods_registered = True


# ---------------------------------------------------------------------------
# Public dispatch surface
# ---------------------------------------------------------------------------


def _resolve(name: str) -> PEFTMethod:
    """Look up a method by name, registering built-ins if needed.

    All public dispatch helpers go through this so a user calling
    ``apply_peft("lora", ...)`` without first invoking
    :func:`ensure_methods_registered` still works — the bootstrap is
    idempotent.
    """
    ensure_methods_registered()
    return PEFT_REGISTRY.get(name)


def apply_peft(model: nn.Module, name: str, **kwargs: Any) -> nn.Module:
    """Apply a registered PEFT method to ``model`` (in-place).

    Args:
        model: The model to adapt. Modified in place by the per-method
            ``apply_*`` function (matches the existing convention).
        name: Registered method name (e.g. ``"lora"``, ``"adalora"``,
            ``"prefix_tuning"``, ``"ia3"``, ``"bitfit"``, ``"adapter"``,
            ``"qlora"``).
        **kwargs: Method-specific kwargs forwarded verbatim to the
            per-method ``apply_*`` function. See each method's
            docstring for accepted kwargs.

    Returns:
        The same ``model`` (chainable).

    Raises:
        ValueError: If ``name`` is not in :data:`PEFT_REGISTRY`.
        TypeError: If the method's wrapper rejects the model shape
            (e.g. Prefix Tuning on a non-MHA base).
    """
    method = _resolve(name)
    return method.apply(model, **kwargs)


def _require_helper(
    method: PEFTMethod,
    helper: str,
    name: str,
) -> Any:
    """Return ``getattr(method, helper)`` or raise NotImplementedError.

    Used for the optional helpers (``merge``, ``unmerge``, ``disable``,
    ``enable``, ``get_parameters``, ``count_parameters``). The error
    message names the method so the user can tell which one rejected
    the call (helpful when iterating over many methods).
    """
    fn = getattr(method, helper)
    if fn is None:
        raise NotImplementedError(
            f"PEFT method '{name}' does not expose a '{helper}' helper. "
            f"Check llm.core.{name} for the per-method equivalent."
        )
    return fn


def get_peft_parameters(model: nn.Module, name: str) -> Iterator[nn.Parameter]:
    """Yield the trainable parameters added by method ``name``.

    Raises:
        NotImplementedError: If the method doesn't expose a parameter
            iterator (callers should fall back to
            ``[p for p in model.parameters() if p.requires_grad]``).
        ValueError: If ``name`` is unknown.
    """
    method = _resolve(name)
    fn = _require_helper(method, "get_parameters", name)
    return fn(model)


def count_peft_parameters(model: nn.Module, name: str) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts for method ``name``.

    Raises:
        NotImplementedError: If the method doesn't expose a count
            helper.
        ValueError: If ``name`` is unknown.
    """
    method = _resolve(name)
    fn = _require_helper(method, "count_parameters", name)
    return fn(model)


def merge_peft(model: nn.Module, name: str) -> nn.Module:
    """Inference-time fold of the adapter into the base weight.

    Raises:
        NotImplementedError: For methods that don't fold (bitfit /
            qlora / prefix_tuning).
        ValueError: If ``name`` is unknown.
    """
    method = _resolve(name)
    fn = _require_helper(method, "merge", name)
    return fn(model)


def unmerge_peft(model: nn.Module, name: str) -> nn.Module:
    """Reverse a previous :func:`merge_peft` call.

    Raises:
        NotImplementedError: For methods that don't expose merge /
            unmerge.
        ValueError: If ``name`` is unknown.
    """
    method = _resolve(name)
    fn = _require_helper(method, "unmerge", name)
    return fn(model)


def disable_peft(model: nn.Module, name: str) -> None:
    """Disable the adapter (e.g. for ablation studies).

    Raises:
        NotImplementedError: For methods that don't expose a disable
            helper (bitfit / qlora / prefix_tuning).
        ValueError: If ``name`` is unknown.
    """
    method = _resolve(name)
    fn = _require_helper(method, "disable", name)
    fn(model)


def enable_peft(model: nn.Module, name: str) -> None:
    """Re-enable a previously disabled adapter.

    Raises:
        NotImplementedError: For methods that don't expose an enable
            helper (bitfit / qlora / prefix_tuning).
        ValueError: If ``name`` is unknown.
    """
    method = _resolve(name)
    fn = _require_helper(method, "enable", name)
    fn(model)


__all__ = [
    "PEFT_REGISTRY",
    "apply_peft",
    "count_peft_parameters",
    "disable_peft",
    "enable_peft",
    "ensure_methods_registered",
    "get_peft_parameters",
    "merge_peft",
    "unmerge_peft",
]
