"""
Quantization module for model compression.

Provides three orthogonal paths:
- Simple post-training quantization (PTQ): INT8/INT4, symmetric/asymmetric,
  per-channel/per-tensor.
- GPTQ (Frantar 2022): Hessian-aware 4-bit/8-bit with packed storage,
  act-order, group_size.
- AWQ (Lin et al., MLSys 2024): activation-aware per-channel scales with
  grid search, then symmetric 4-bit/8-bit packed storage.

All paths share calibration infrastructure via
:class:`CalibrationDataCollector`.
"""

# AWQ path
from llm.quantization._awq_layer import AWQQuantizedLinear

# GPTQ path
from llm.quantization._gptq_layer import GPTQQuantizedLinear
from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies
from llm.quantization.awq import (
    AWQConfig,
    AWQQuantizer,
    quantize_model_awq,
    quantize_model_awq_with_collector,
)
from llm.quantization.calibration import ActivationStats, CalibrationDataCollector
from llm.quantization.gptq import (
    GPTQConfig,
    GPTQQuantizer,
    quantize_model_gptq,
    quantize_model_with_collector,
)

# Simple PTQ path
from llm.quantization.ptq import (
    QuantConfig,
    QuantizedLinear,
    compute_model_size,
    quantize_linear_layer,
    quantize_model,
)

__all__ = [
    "AWQConfig",
    "AWQQuantizedLinear",
    "AWQQuantizer",
    "ActivationStats",
    "CalibrationDataCollector",
    "GPTQConfig",
    "GPTQQuantizedLinear",
    "GPTQQuantizer",
    "LayerQuantPolicy",
    "QuantConfig",
    "QuantizedLinear",
    "compute_model_size",
    "quantize_linear_layer",
    "quantize_model",
    "quantize_model_awq",
    "quantize_model_awq_with_collector",
    "quantize_model_gptq",
    "quantize_model_with_collector",
    "resolve_layer_policies",
]
