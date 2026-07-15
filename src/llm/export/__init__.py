"""Model export utilities.

Public surface:
    - ``export_to_onnx`` / ``verify_onnx`` / ``get_onnx_info`` — the
      ONNX reference implementation (preserved as a stable API).
    - :data:`EXPORT_REGISTRY` and :func:`export_model` — the
      registry-driven dispatch for any export target (built-in
      ``onnx`` plus third-party plugins via the
      ``llm.export_backends`` entry-point group).
"""

from llm.export.onnx import export_to_onnx, get_onnx_info, verify_onnx
from llm.export.registry import (
    EXPORT_REGISTRY,
    ExportBackendFactory,
    build_onnx_exporter,
    ensure_exporters_registered,
    export_model,
)

__all__ = [
    "EXPORT_REGISTRY",
    "ExportBackendFactory",
    "build_onnx_exporter",
    "ensure_exporters_registered",
    "export_model",
    "export_to_onnx",
    "get_onnx_info",
    "verify_onnx",
]
