"""Model export utilities.

Public surface:
    - ``export_to_onnx`` / ``verify_onnx`` / ``get_onnx_info`` — the
      ONNX reference implementation (preserved as a stable API).
    - ``export_to_torchscript`` — the TorchScript export target
      (Tier 3 #11). Registered via the ``llm.export_backends``
      setuptools entry point.
    - ``export_to_gguf`` / ``build_gguf_exporter`` — the GGUF export
      target (ADR-011): GGUF v3 container with F16/F32/Q4_0/Q8_0 tensor
      types. Registered via the ``llm.export_backends`` setuptools
      entry point.
    - ``load_gguf_model`` — rebuild a model from a GGUF the exporter
      wrote with ``model_config=`` (round-71 load-back milestone).
    - :data:`EXPORT_REGISTRY` and :func:`export_model` — the
      registry-driven dispatch for any export target (built-in
      ``onnx`` plus third-party plugins via the
      ``llm.export_backends`` entry-point group).
"""

from llm.export.gguf import build_gguf_exporter, export_to_gguf, load_gguf_model
from llm.export.onnx import export_to_onnx, get_onnx_info, verify_onnx
from llm.export.registry import (
    EXPORT_REGISTRY,
    ExportBackendFactory,
    build_onnx_exporter,
    ensure_exporters_registered,
    export_model,
)
from llm.export.torchscript import build_torchscript_exporter, export_to_torchscript

__all__ = [
    "EXPORT_REGISTRY",
    "ExportBackendFactory",
    "build_gguf_exporter",
    "build_onnx_exporter",
    "build_torchscript_exporter",
    "ensure_exporters_registered",
    "export_model",
    "export_to_gguf",
    "export_to_onnx",
    "export_to_torchscript",
    "get_onnx_info",
    "load_gguf_model",
    "verify_onnx",
]
