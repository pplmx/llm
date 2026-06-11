"""Load models and tokenizers for inference serving."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from llm.models.decoder import DecoderModel
from llm.runtime import ModelFactory
from llm.runtime.tokenizer_factory import TokenizerFactory
from llm.serving.config import ServingConfig
from llm.training.core.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Minimal view of a training checkpoint file."""

    path: Path
    model_state: dict[str, torch.Tensor]
    model_config: dict[str, Any] | None = None
    epoch: int | None = None
    loss: float | None = None


def load_training_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> TrainingCheckpoint:
    """Load a training checkpoint produced by CheckpointManager."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if "model_state" not in payload:
        raise ValueError(f"Checkpoint missing 'model_state': {ckpt_path}")

    return TrainingCheckpoint(
        path=ckpt_path,
        model_state=payload["model_state"],
        model_config=payload.get("model_config"),
        epoch=payload.get("epoch"),
        loss=payload.get("loss"),
    )


def infer_vocab_size(state_dict: dict[str, torch.Tensor]) -> int:
    """Infer vocabulary size from an LM head or embedding weight tensor."""
    if "lm_head.weight" in state_dict:
        return int(state_dict["lm_head.weight"].shape[0])
    if "embedding.token_embedding.weight" in state_dict:
        return int(state_dict["embedding.token_embedding.weight"].shape[0])
    raise ValueError("Cannot infer vocab_size from checkpoint state dict")


def infer_num_layers(state_dict: dict[str, torch.Tensor]) -> int | None:
    """Count transformer blocks present in a state dict."""
    indices: set[int] = set()
    prefix = "transformer_blocks."
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        index_str, _, _ = rest.partition(".")
        if index_str.isdigit():
            indices.add(int(index_str))
    return max(indices) + 1 if indices else None


def _build_decoder(
    *,
    serving_config: ServingConfig,
    model_config: dict[str, Any] | None,
    state_dict: dict[str, torch.Tensor],
) -> DecoderModel:
    if model_config is not None:
        cfg = ModelConfig.model_validate(model_config)
        model = ModelFactory.from_config(cfg)
    else:
        vocab_size = infer_vocab_size(state_dict)
        num_layers = infer_num_layers(state_dict) or serving_config.num_layers
        logger.warning(
            "Checkpoint has no model_config; using ServingConfig architecture with inferred vocab_size=%s",
            vocab_size,
        )
        model = ModelFactory.build(
            "decoder",
            vocab_size=vocab_size,
            hidden_size=serving_config.hidden_size,
            num_layers=num_layers,
            num_heads=serving_config.num_heads,
            max_seq_len=serving_config.max_seq_len,
            num_kv_heads=serving_config.num_kv_heads,
            attn_impl=serving_config.attn_impl,
            mlp_impl=serving_config.mlp_impl,
            use_moe=serving_config.use_moe,
            num_experts=serving_config.num_experts,
            top_k=serving_config.top_k,
            use_glu=False,
            intermediate_size=None,
            embedding_dropout_p=0.0,
            attn_dropout_p=0.0,
            mlp_dropout_p=0.0,
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected[:5])
    return model


def load_tokenizer(config: ServingConfig) -> Any:
    """Load tokenizer from config or fall back to a printable character tokenizer."""
    return TokenizerFactory.from_serving_config(config)


def load_model_and_tokenizer(config: ServingConfig) -> tuple[DecoderModel, Any]:
    """Build model/tokenizer for serving, loading weights when model_path is set."""
    if not config.model_path:
        return _create_dummy_model_and_tokenizer(config)

    checkpoint = load_training_checkpoint(config.model_path)
    model = _build_decoder(
        serving_config=config,
        model_config=checkpoint.model_config,
        state_dict=checkpoint.model_state,
    )
    tokenizer = load_tokenizer(config)
    return model, tokenizer


def _create_dummy_model_and_tokenizer(config: ServingConfig) -> tuple[DecoderModel, Any]:
    """Fallback demo model when no checkpoint is configured."""
    import string

    from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

    tokenizer = SimpleCharacterTokenizer([string.printable])
    model = ModelFactory.build(
        "decoder",
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        num_kv_heads=config.num_kv_heads,
        attn_impl=config.attn_impl,
        mlp_impl=config.mlp_impl,
    )
    return model, tokenizer
