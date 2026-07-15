"""TorchScript export backend.

This is the second target of :data:`llm.export.registry.EXPORT_REGISTRY`
and the first one to register through the ``llm.export_backends``
setuptools entry-point group (rather than the in-code registration
used by the built-in ``onnx`` target). See
:func:`llm.export._plugins.register_torchscript_exporter` for the
hook the entry point points at.

TorchScript ships with PyTorch, so this backend adds no runtime
dependencies. The exported artifact is a ``.pt`` file loadable via
``torch.jit.load`` — useful for deployment paths that can't or
won't bring up an ONNX runtime.

Two export modes are supported:

- ``method='trace'`` (default): records operations with example
  inputs. Works for any model that is ``forward``-passable with
  static shapes. The cache wrapper forces ``use_cache=False`` so
  the tracer doesn't record KV-cache branching.
- ``method='script'``: compiles the model with the TorchScript
  compiler. Requires a model whose forward uses only
  TorchScript-supported constructs. Models with dynamic Python
  control flow may not script; in that case, fall back to
  ``trace``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from llm.export._wrapper import ExportCacheWrapper

if TYPE_CHECKING:
    pass


def export_to_torchscript(
    model: nn.Module,
    output_path: str | Path,
    *,
    method: str = "trace",
    input_shape: tuple[int, int] = (1, 32),
    example_inputs: torch.Tensor | None = None,
    strict: bool = True,
    **kwargs: Any,
) -> Path:
    """Export a model to TorchScript.

    Args:
        model: The model to export.
        output_path: Path to write the ``.pt`` artifact to. Parent
            directories are created automatically.
        method: ``"trace"`` (default) or ``"script"``.
        input_shape: ``(batch_size, seq_len)`` for the dummy input
            used by ``trace``. Ignored if ``example_inputs`` is
            provided.
        example_inputs: Pre-built dummy tensor. Overrides
            ``input_shape`` when supplied.
        strict: Forwarded to :func:`torch.jit.trace`. When ``False``
            the tracer silently records missing ops instead of
            raising. Default ``True``.
        **kwargs: Forwarded to :func:`torch.jit.trace` or
            :func:`torch.jit.script`.

    Returns:
        The resolved output path.

    Raises:
        ValueError: If ``method`` is not ``"trace"`` or ``"script"``.
    """
    if method not in {"trace", "script"}:
        raise ValueError(f"method must be 'trace' or 'script', got {method!r}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    wrapped = ExportCacheWrapper(model)
    wrapped.eval()

    if method == "trace":
        if example_inputs is None:
            batch_size, seq_len = input_shape
            device = next(model.parameters()).device
            example_inputs = torch.randint(0, 100, (batch_size, seq_len), device=device)
        scripted = torch.jit.trace(wrapped, example_inputs, strict=strict, **kwargs)
    else:  # method == "script"
        scripted = torch.jit.script(wrapped, **kwargs)

    torch.jit.save(scripted, str(output_path))
    return output_path


def build_torchscript_exporter(
    model: nn.Module,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Factory for the TorchScript export target.

    Thin wrapper over :func:`export_to_torchscript` so the registry
    contract (``(model, output_path, **kwargs) -> Path``) matches
    every other target.
    """
    return export_to_torchscript(model, output_path, **kwargs)


__all__ = ["build_torchscript_exporter", "export_to_torchscript"]
