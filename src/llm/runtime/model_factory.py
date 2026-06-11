"""Central model construction for training, serving, and compat loaders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from llm.models.decoder import DecoderModel
from llm.runtime.registry import Registry
from llm.training.core.config import ModelConfig

ModelBuilder = Callable[..., nn.Module]

MODEL_REGISTRY: Registry[ModelBuilder] = Registry("Model")


def build_decoder(
    *,
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    max_seq_len: int = 512,
    intermediate_size: int | None = None,
    embedding_dropout_p: float = 0.1,
    attn_dropout_p: float = 0.1,
    mlp_dropout_p: float = 0.1,
    num_experts: int = 0,
    top_k: int = 0,
    num_kv_heads: int | None = None,
    use_glu: bool = False,
    attn_impl: str = "mha",
    mlp_impl: str = "mlp",
    norm_eps: float = 1e-5,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> DecoderModel:
    """Construct a DecoderModel from explicit architecture kwargs."""
    return DecoderModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        intermediate_size=intermediate_size,
        embedding_dropout_p=embedding_dropout_p,
        attn_dropout_p=attn_dropout_p,
        mlp_dropout_p=mlp_dropout_p,
        num_experts=num_experts,
        top_k=top_k,
        num_kv_heads=num_kv_heads,
        use_glu=use_glu,
        attn_impl=attn_impl,
        mlp_impl=mlp_impl,
        norm_eps=norm_eps,
        device=device,
        dtype=dtype,
        **kwargs,
    )


def decoder_kwargs_from_config(config: ModelConfig, **overrides: Any) -> dict[str, Any]:
    """Map a ModelConfig into DecoderModel constructor kwargs."""
    kwargs: dict[str, Any] = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "max_seq_len": config.max_seq_len,
        "intermediate_size": config.intermediate_size,
        "embedding_dropout_p": config.dropout,
        "attn_dropout_p": config.dropout,
        "mlp_dropout_p": config.dropout,
        "num_experts": config.num_experts,
        "top_k": config.top_k,
        "num_kv_heads": config.num_kv_heads,
        "use_glu": config.use_glu,
        "attn_impl": config.attn_impl,
        "mlp_impl": config.mlp_impl,
    }
    kwargs.update(overrides)
    return kwargs


class ModelFactory:
    """Resolve registered model builders from typed or raw configuration."""

    @staticmethod
    def from_config(config: ModelConfig, *, model_type: str = "decoder", **overrides: Any) -> nn.Module:
        kwargs = decoder_kwargs_from_config(config, **overrides)
        return ModelFactory.build(model_type, **kwargs)

    @staticmethod
    def build(model_type: str = "decoder", **kwargs: Any) -> nn.Module:
        builder = MODEL_REGISTRY.get(model_type)
        return builder(**kwargs)
