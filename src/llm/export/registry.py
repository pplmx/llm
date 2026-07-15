"""Export backend registry and bootstrap.

Mirrors the ``generation/registry.py`` pattern so third-party
export targets (e.g. ``torch.compile``, ``vLLM``, ``TensorRT-LLM``,
``torch.export``, ``OpenVINO``) can plug in via the
``llm.export_backends`` setuptools entry-point group without
forking ``export/``.

Built-in targets:
    ``onnx`` — wraps the existing ``export_to_onnx`` function. This
    is the canonical reference implementation; the entry-point load
    raises if a plugin claims the same name, which is intentional
    (the built-in is the source of truth).

Usage:
    >>> from llm.export.registry import export_model
    >>> export_model("onnx", model, "out.onnx", input_shape=(1, 32))

Plugin authors can register a target via ``pyproject.toml``:

    [project.entry-points."llm.export_backends"]
    my_target = "my_pkg.exporters:build_my_target"
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm.runtime.plugins import load_entry_point_registry
from llm.runtime.registry import Registry

if TYPE_CHECKING:
    import torch.nn as nn

# A factory receives the model, an output path, and any
# target-specific kwargs; it must write the artifact and return
# the resolved path. This shape is intentionally narrow — anything
# richer (e.g. a class with ``.export()``) is easy to wrap.
ExportBackendFactory = Callable[..., Path]

EXPORT_REGISTRY: Registry[ExportBackendFactory] = Registry("ExportBackend")

_exporters_registered = False


def build_onnx_exporter(
    model: nn.Module,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Factory for the built-in ONNX export target.

    Thin wrapper over :func:`llm.export.onnx.export_to_onnx` so the
    registry contract (``(model, output_path, **kwargs) -> Path``)
    matches every other target. The wrapper exists purely so the
    registry doesn't have to special-case keyword forwarding for
    ONNX's wider surface (``opset_version``, ``dynamic_axes``,
    ``verbose``, ...).
    """
    from llm.export.onnx import export_to_onnx

    return export_to_onnx(model, output_path, **kwargs)


def ensure_exporters_registered() -> None:
    """Idempotently register built-in exporters and load entry points.

    Built-ins are registered BEFORE the entry-point load so a plugin
    that claims ``onnx`` raises loudly — the built-in is the
    reference implementation. This matches the convention in
    ``generation/registry.ensure_backends_registered``.
    """
    global _exporters_registered
    if _exporters_registered:
        return

    EXPORT_REGISTRY.register("onnx", build_onnx_exporter)
    load_entry_point_registry("llm.export_backends", EXPORT_REGISTRY)
    _exporters_registered = True


def export_model(
    name: str,
    model: nn.Module,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Resolve a registered export target and run it.

    Args:
        name: Registered export target name (e.g. ``"onnx"``).
        model: The model to export.
        output_path: Where to write the artifact.
        **kwargs: Target-specific kwargs forwarded to the factory.

    Returns:
        The resolved output path.

    Raises:
        ValueError: If ``name`` is not in :data:`EXPORT_REGISTRY`.
    """
    ensure_exporters_registered()
    return EXPORT_REGISTRY.get(name)(model=model, output_path=output_path, **kwargs)


__all__ = [
    "EXPORT_REGISTRY",
    "ExportBackendFactory",
    "build_onnx_exporter",
    "ensure_exporters_registered",
    "export_model",
]
