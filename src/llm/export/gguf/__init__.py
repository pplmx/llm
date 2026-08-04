"""GGUF model format module (ADR-011).

Implements the GGUF v3 container — header / typed metadata / tensor
info plus reader and writer — and the two GGML block-quantization
schemes shipped in v1 (Q4_0 and Q8_0), then exposes the GGUF export
target for :data:`llm.export.registry.EXPORT_REGISTRY`.

Public surface:

- format: :class:`GGUFHeader`, :class:`GGUFTensorInfo`,
  :class:`GGUFValueType`, :class:`GGMLQuantizationType`,
  :class:`GGUFError`, and the ``GGUF_*`` constants;
- I/O: :class:`GGUFWriter` / :class:`GGUFReader`;
- quantization: :func:`quantize_q4_0` / :func:`dequantize_q4_0` and
  :func:`quantize_q8_0` / :func:`dequantize_q8_0`;
- export: :func:`export_to_gguf` / :func:`build_gguf_exporter`.
"""

from llm.export.gguf.exporter import build_gguf_exporter, export_to_gguf
from llm.export.gguf.quant import (
    dequantize_q4_0,
    dequantize_q8_0,
    quantize_q4_0,
    quantize_q8_0,
)
from llm.export.gguf.reader import GGUFReader
from llm.export.gguf.spec import (
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_HEADER_SIZE,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGMLQuantizationType,
    GGUFError,
    GGUFHeader,
    GGUFTensorInfo,
    GGUFValueType,
)
from llm.export.gguf.writer import GGUFWriter

__all__ = [
    "GGUF_DEFAULT_ALIGNMENT",
    "GGUF_HEADER_SIZE",
    "GGUF_MAGIC",
    "GGUF_VERSION",
    "GGMLQuantizationType",
    "GGUFError",
    "GGUFHeader",
    "GGUFReader",
    "GGUFTensorInfo",
    "GGUFValueType",
    "GGUFWriter",
    "build_gguf_exporter",
    "dequantize_q4_0",
    "dequantize_q8_0",
    "export_to_gguf",
    "quantize_q4_0",
    "quantize_q8_0",
]
