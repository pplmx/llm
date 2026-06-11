"""Component registries backed by runtime.Registry."""

from __future__ import annotations

import torch.nn as nn

from llm.core.rms_norm import RMSNorm
from llm.runtime.registry import Registry, decorator_register

ATTENTION_REGISTRY: Registry[type] = Registry("Attention")
MLP_REGISTRY: Registry[type] = Registry("MLP")
NORM_REGISTRY: Registry[type] = Registry("Normalization")

register_attention = decorator_register(ATTENTION_REGISTRY)
register_mlp = decorator_register(MLP_REGISTRY)


def ensure_norms_registered() -> None:
    if "layer_norm" not in NORM_REGISTRY:
        NORM_REGISTRY.register("layer_norm", nn.LayerNorm)
    if "rms_norm" not in NORM_REGISTRY:
        NORM_REGISTRY.register("rms_norm", RMSNorm)
