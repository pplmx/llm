"""Component registries backed by runtime.Registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from llm.core.rms_norm import RMSNorm
from llm.runtime.registry import Registry, decorator_register

ATTENTION_REGISTRY: Registry[type] = Registry("Attention")
MLP_REGISTRY: Registry[type] = Registry("MLP")
# NORM_REGISTRY stores factory callables ``(**kwargs) -> nn.Module`` rather
# than the module class itself. This matches MODEL_REGISTRY (which also
# stores callables) and lets future norms do shape inference or context-
# aware construction without changing the registry contract.
NORM_REGISTRY: Registry[Callable[..., nn.Module]] = Registry("Normalization")

register_attention = decorator_register(ATTENTION_REGISTRY)
register_mlp = decorator_register(MLP_REGISTRY)


# Per-attention capability map, kept in sync with each ``@register_attention(...)``
# call site. ``ModelConfig.check_consistency`` consults this to fail fast on
# combinations that cannot work (e.g., MLA + KV cache).
ATTENTION_KV_CACHE_CAPABILITY: dict[str, bool] = {
    # ``mha`` supports KV cache (registered in ``core.attn.mha``).
    "mha": True,
    # ``mla`` does NOT support KV cache yet — see ``core.attn.mla``.
    "mla": False,
}


def set_attention_kv_cache_capability(name: str, supports: bool) -> None:
    """Record whether ``name`` supports KV cache.

    Called from each attention implementation at import time, alongside
    ``@register_attention``. Validation in ``ModelConfig.check_consistency``
    raises if a model declares ``attn_impl`` that has no capability record.
    """
    ATTENTION_KV_CACHE_CAPABILITY[name] = supports


def attention_supports_kv_cache(name: str) -> bool:
    """Return whether the registered attention impl supports KV cache.

    Raises ``KeyError`` if the impl has not declared its capability — that is
    a registration bug, not a user error. ``ModelConfig`` validates this at
    config-load time.
    """
    return ATTENTION_KV_CACHE_CAPABILITY[name]


def _make_layer_norm(*args: Any, **kwargs: Any) -> nn.Module:
    """Factory for ``nn.LayerNorm`` accepting both positional and keyword args.

    The caller pattern from ``TransformerBlock`` is
    ``factory(hidden_size, eps=norm_eps, **factory_kwargs)``, where the
    first positional is ``normalized_shape``.
    """
    return nn.LayerNorm(*args, **kwargs)


def _make_rms_norm(*args: Any, **kwargs: Any) -> nn.Module:
    """Factory for ``RMSNorm`` accepting both positional and keyword args."""
    return RMSNorm(*args, **kwargs)


def ensure_norms_registered() -> None:
    if "layer_norm" not in NORM_REGISTRY:
        NORM_REGISTRY.register("layer_norm", _make_layer_norm)
    if "rms_norm" not in NORM_REGISTRY:
        NORM_REGISTRY.register("rms_norm", _make_rms_norm)
