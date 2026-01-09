"""
Quantization module for model compression.

Provides tools for post-training quantization (PTQ) and calibration.
"""

from llm.quantization.calibration import ActivationStats, CalibrationDataCollector
from llm.quantization.ptq import (
    QuantConfig,
    quantize_linear_layer,
    quantize_model,
)

__all__ = [
    "ActivationStats",
    "CalibrationDataCollector",
    "QuantConfig",
    "quantize_linear_layer",
    "quantize_model",
]
