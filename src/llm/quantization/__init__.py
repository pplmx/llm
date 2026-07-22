"""
Quantization module for model compression.

Provides two orthogonal paths:
- Simple post-training quantization (PTQ): INT8/INT4, symmetric/asymmetric,
  per-channel/per-tensor.
- GPTQ (Frantar 2022): Hessian-aware 4-bit/8-bit with packed storage,
  act-order, group_size.

Both share calibration infrastructure via :class:`CalibrationDataCollector`.
"""

from llm.quantization.calibration import ActivationStats, CalibrationDataCollector

# Simple PTQ path
from llm.quantization.ptq import (
    QuantConfig,
    QuantizedLinear,
    compute_model_size,
    quantize_linear_layer,
    quantize_model,
)

# GPTQ path
from llm.quantization._gptq_layer import GPTQQuantizedLinear
from llm.quantization.gptq import (
    GPTQConfig,
    GPTQQuantizer,
    quantize_model_gptq,
    quantize_model_with_collector,
)

__all__ = [
    # Calibration
    "ActivationStats",
    "CalibrationDataCollector",
    # Simple PTQ
    "QuantConfig",
    "QuantizedLinear",
    "compute_model_size",
    "quantize_linear_layer",
    "quantize_model",
    # GPTQ
    "GPTQConfig",
    "GPTQQuantizer",
    "GPTQQuantizedLinear",
    "quantize_model_gptq",
    "quantize_model_with_collector",
]
