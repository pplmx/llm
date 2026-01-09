"""
HuggingFace Compatibility module.

Provides utilities for loading and converting HuggingFace models.
"""

from llm.compat.hf_loader import from_pretrained, list_supported_architectures
from llm.compat.weight_mapping import (
    ARCHITECTURE_MAPPINGS,
    convert_hf_weights,
    get_weight_mapping,
)

__all__ = [
    "ARCHITECTURE_MAPPINGS",
    "convert_hf_weights",
    "from_pretrained",
    "get_weight_mapping",
    "list_supported_architectures",
]
