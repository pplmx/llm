"""PEFT adapter-only checkpoint save/load (T2 PEFT #47).

Saves ONLY the trainable adapter parameters added by a PEFT method —
not the full model state — so adapters can be shared across runs
(across checkpoints, across base models, across teams) without
copying the (usually huge) base weights every time.

Storage format (``torch.save``):

    {
        "format_version": PEFT_CHECKPOINT_FORMAT_VERSION,  # "1.0"
        "method_name": "lora",
        "peft_kwargs": {"rank": 8, "alpha": 16.0},  # informational
        "state_dict": {
            # positional keys: f"{method_name}.{idx}" for each adapter param
            "lora.0": tensor,
            "lora.1": tensor,
            ...
        },
    }

The keys are positional because the structural identity of adapter
parameters is unstable across processes (``id()`` changes), but the
ORDER of :func:`PEFTMethod.get_parameters` output is deterministic
for the same model architecture + same apply kwargs. Loading matches
by position: saved tensor at index ``i`` lands in the model's
adapter parameter at index ``i``.

The ``peft_kwargs`` dict is informational — :func:`load_peft` uses
it to re-apply the method when the model hasn't been wrapped yet
(common case for adapter sharing). The user can override individual
kwargs via :func:`load_peft`'s ``**override_kwargs``.

Forward compatibility: bumping :data:`PEFT_CHECKPOINT_FORMAT_VERSION`
is the supported migration path. :func:`load_peft` rejects unknown
versions with a loud :class:`ValueError`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from llm.core.peft.registry import _resolve

if TYPE_CHECKING:
    import torch.nn as nn


#: On-disk format version. Bump when the payload shape changes
#: incompatibly; ``load_peft`` rejects unknown versions.
PEFT_CHECKPOINT_FORMAT_VERSION = "1.0"


def _collect_adapter_params(
    model: nn.Module,
    method_name: str,
) -> list[nn.Parameter]:
    """Collect the method's adapter parameters in deterministic order.

    The output order matches what :func:`PEFTMethod.get_parameters`
    yields — typically ``model.modules()`` iteration order, which is
    deterministic for the same architecture + same apply kwargs.
    Loading matches by position so this order MUST be stable.
    """
    method = _resolve(method_name)
    if method.get_parameters is None:
        raise NotImplementedError(
            f"PEFT method '{method_name}' does not expose get_parameters. "
            f"Check llm.core.{method_name} for the per-method equivalent."
        )
    return list(method.get_parameters(model))


def save_peft(
    model: nn.Module,
    path: str | Path,
    method_name: str,
    **peft_kwargs: Any,
) -> Path:
    """Save only the adapter parameters added by ``method_name``.

    Writes a single ``torch.save``-compatible file containing:

    - ``format_version``: :data:`PEFT_CHECKPOINT_FORMAT_VERSION`
    - ``method_name``: registered name (e.g. ``"lora"``)
    - ``peft_kwargs``: kwargs the caller used (informational; used by
      :func:`load_peft` to re-apply the method on a fresh model)
    - ``state_dict``: the adapter parameters, keyed by position

    Args:
        model: PEFT-applied model. Must have ``method_name`` already
            applied (use :func:`apply_peft` first).
        path: Destination path. Parent directories are created if
            they don't exist.
        method_name: Registered method name (e.g. ``"lora"``,
            ``"adapter"``, ``"bitfit"``).
        **peft_kwargs: Method-specific kwargs — stored in the
            metadata envelope so :func:`load_peft` can re-apply the
            method automatically when the destination model is fresh.

    Returns:
        The resolved ``Path`` the file was written to.

    Raises:
        ValueError: If ``method_name`` is not in the registry.
        NotImplementedError: If the method doesn't expose
            ``get_parameters``.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    params = _collect_adapter_params(model, method_name)

    payload: dict[str, Any] = {
        "format_version": PEFT_CHECKPOINT_FORMAT_VERSION,
        "method_name": method_name,
        "peft_kwargs": dict(peft_kwargs),
        "state_dict": {
            f"{method_name}.{i}": p.detach().cpu().clone()
            for i, p in enumerate(params)
        },
    }
    torch.save(payload, out_path)
    return out_path


def load_peft(
    model: nn.Module,
    path: str | Path,
    method_name: str,
    **override_kwargs: Any,
) -> nn.Module:
    """Load adapter parameters from ``path`` into ``model``.

    If the model hasn't had ``method_name`` applied yet (no wrappers
    of the expected type), :func:`apply_peft` is called first using
    the kwargs stored in the checkpoint — caller-supplied
    ``override_kwargs`` take precedence over the saved kwargs.

    Args:
        model: Destination model. If PEFT is not yet applied, it is
            applied automatically using the checkpoint's saved
            kwargs (overridable via ``override_kwargs``).
        path: Path to a file written by :func:`save_peft`.
        method_name: Expected method name — must match the
            ``method_name`` field in the checkpoint.
        **override_kwargs: Override individual ``peft_kwargs`` from
            the checkpoint (e.g. ``rank=16`` to widen an adapter when
            loading). Useful for adapter-surgery workflows.

    Returns:
        The same ``model`` with the adapter parameters loaded
        byte-identically (chainable).

    Raises:
        FileNotFoundError: If ``path`` doesn't exist.
        ValueError: If the method name, format version, or parameter
            count doesn't match expectations.
        RuntimeError: If the model's adapter parameter count
            doesn't match the checkpoint (after re-applying if
            needed) — usually a sign the destination architecture
            differs from the source.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"PEFT checkpoint not found: {in_path}")

    # Resolve the method FIRST so an unknown method_name raises a
    # clear "not found" ValueError (matching the apply / save error
    # semantics) before we even look at the on-disk payload.
    method = _resolve(method_name)

    payload = torch.load(in_path, weights_only=False, map_location="cpu")

    # Format-version check. Bumping the version is the supported
    # migration path; unknown versions get a loud rejection so users
    # don't silently corrupt state.
    fmt_version = payload.get("format_version")
    if fmt_version != PEFT_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported PEFT checkpoint format_version={fmt_version!r} "
            f"(expected {PEFT_CHECKPOINT_FORMAT_VERSION!r}). "
            f"Bump llm.core.peft.checkpoint.PEFT_CHECKPOINT_FORMAT_VERSION "
            f"or upgrade llm."
        )

    # Method-name check. Catches the "saved as LoRA, loading as IA3"
    # mistake early — without this, the param-count check below
    # would also fire, but with a less informative message.
    saved_method = payload.get("method_name")
    if saved_method != method_name:
        raise ValueError(
            f"PEFT checkpoint method name mismatch: "
            f"checkpoint says {saved_method!r}, requested {method_name!r}. "
            f"Refusing to load — pass method_name={saved_method!r} explicitly "
            f"if you really mean to load this checkpoint."
        )

    # Re-apply the method if the model doesn't already have it. This
    # is the common case for cross-run / cross-model adapter sharing.
    is_applied_fn = method.is_applied
    if is_applied_fn is None or not is_applied_fn(model):
        peft_kwargs = {**payload.get("peft_kwargs", {}), **override_kwargs}
        method.apply(model, **peft_kwargs)

    # Now the model has fresh adapter parameters. Copy the saved
    # tensors into them, matched by position.
    current = _collect_adapter_params(model, method_name)
    saved: dict[str, torch.Tensor] = payload["state_dict"]

    if len(current) != len(saved):
        raise RuntimeError(
            f"PEFT adapter param count mismatch for method "
            f"{method_name!r}: model has {len(current)} adapter "
            f"parameters after re-apply, checkpoint has "
            f"{len(saved)}. The destination architecture likely "
            f"differs from the source."
        )

    with torch.no_grad():
        for i, p in enumerate(current):
            saved_tensor = saved[f"{method_name}.{i}"]
            # Cast to the live parameter's device + dtype to handle
            # cross-device / cross-dtype transfers cleanly (e.g.
            # load a CPU checkpoint into a CUDA model — common for
            # adapter sharing).
            p.data.copy_(saved_tensor.to(p.device, dtype=p.dtype))

    return model


__all__ = [
    "PEFT_CHECKPOINT_FORMAT_VERSION",
    "load_peft",
    "save_peft",
]
