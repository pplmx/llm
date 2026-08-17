"""Load models and tokenizers for inference serving."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from llm.models.decoder import DecoderModel
from llm.runtime.model_factory import ModelFactory
from llm.runtime.tokenizer_factory import TokenizerFactory
from llm.serving.config import ServingConfig
from llm.training.core.checkpoint import load_checkpoint_payload
from llm.training.core.config import ModelConfig
from llm.utils.serialization import (
    register_framework_safe_globals as _register_framework_safe_globals,
)

logger = logging.getLogger(__name__)


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize torch.compile state-dict keys to the plain module namespace.

    Checkpoints saved before the save-side fix may carry ``_orig_mod.``
    prefixes (``torch.compile`` renames every key). ``llm-serve`` builds a
    plain ``DecoderModel``, so the prefix must be stripped before
    ``load_state_dict`` or every weight is silently dropped.
    """
    if any(key.startswith("_orig_mod.") for key in state_dict):
        return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    return state_dict


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Minimal view of a training checkpoint file."""

    path: Path
    model_state: dict[str, torch.Tensor]
    model_config: dict[str, Any] | None = None
    epoch: int | None = None
    loss: float | None = None
    model_obj: torch.nn.Module | None = None


def load_training_checkpoint(path: str | Path) -> TrainingCheckpoint:
    """Load a training checkpoint produced by CheckpointManager.

    Accepts both the v2 split layout (``<stem>.safetensors`` +
    ``<stem>.meta.json`` + ``<stem>.extra_state.pt``, the modern
    ``CheckpointManager`` format) and the legacy v0.0.5 single-file
    ``.pt`` blob, plus the bare ``nn.Module`` pickle emitted by
    ``llm-quantize`` (a quantized model with ``GPTQQuantizedLinear`` /
    ``AWQQuantizedLinear`` / ``SmoothQuantLinear`` submodules whose
    per-layer quantization parameters live on the module instances).
    ``path`` may be a stem, any of the three v2 sidecar paths, or a
    legacy ``.pt`` path.
    """
    ckpt_path = Path(path)
    # A bare-module blob (e.g. ``llm-quantize --output model.pt``) is not a
    # checkpoint dict: detect it before the layout resolution below so it is
    # not routed through the legacy path (which would emit a misleading
    # "run llm-migrate-ckpt" DeprecationWarning for a quantized artifact).
    #
    # Security (RIL ISS-170): the probe loads with ``weights_only=True`` so an
    # untrusted ``.pt`` cannot execute arbitrary ``__reduce__`` code at serve
    # time. The framework's own module classes are allowlisted (see
    # ``_register_framework_safe_globals``), so legitimate quantized blobs load
    # normally; a pickle referencing anything outside the allowlist raises
    # UnpicklingError and is REFUSED rather than silently re-loaded with
    # ``weights_only=False`` (matching the already-hardened PEFT sidecar path).
    if ckpt_path.exists() and ckpt_path.suffix == ".pt":
        _register_framework_safe_globals()
        try:
            obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - probe only; any failure falls through
            # to the checkpoint-layout resolution below, which raises the
            # proper format-specific error for this path.
            obj = None
        if isinstance(obj, torch.nn.Module):
            logger.info("Loading bare quantized model blob from %s", ckpt_path)
            return TrainingCheckpoint(
                path=ckpt_path,
                model_state=obj.state_dict(),
                model_obj=obj,
            )
    payload = load_checkpoint_payload(ckpt_path)
    if payload is None:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} (checked the legacy .pt and the v2 split layout)")
    if "model_state" not in payload:
        raise ValueError(f"Checkpoint missing 'model_state': {ckpt_path}")

    return TrainingCheckpoint(
        path=ckpt_path,
        model_state=_strip_compile_prefix(payload["model_state"]),
        model_config=payload.get("model_config"),
        epoch=payload.get("epoch"),
        loss=payload.get("loss"),
    )


def infer_vocab_size(state_dict: dict[str, torch.Tensor]) -> int:
    """Infer vocabulary size from an LM head or embedding weight tensor.

    The embedding fallback key was a misspelling (``embedding.token_embedding``
    vs the real ``embedding_layer.token_embeddings``), making the branch
    unreachable — a config-less **tied-embedding** checkpoint (no
    ``lm_head.weight``, per ``llm.compat.weight_mapping``) wrongly raised
    ``Cannot infer vocab_size`` even though the embedding tensor was present
    and loadable (RIL ISS-167).
    """
    if "lm_head.weight" in state_dict:
        return int(state_dict["lm_head.weight"].shape[0])
    if "embedding_layer.token_embeddings.weight" in state_dict:
        return int(state_dict["embedding_layer.token_embeddings.weight"].shape[0])
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
            num_experts=serving_config.num_experts,
            top_k=serving_config.top_k,
            use_glu=False,
            intermediate_size=None,
            embedding_dropout_p=0.0,
            attn_dropout_p=0.0,
            mlp_dropout_p=0.0,
        )

    if not isinstance(model, DecoderModel):
        raise TypeError(f"checkpoint model must be a DecoderModel, got {type(model).__name__}")

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
    """Build model/tokenizer for serving, loading weights when model_path is set.

    PEFT integration (T2 PEFT #49): if ``config.peft_method`` is set, the
    loader applies the method to the freshly loaded base model and
    (optionally) loads the sidecar from ``config.peft_adapter_path``.
    Without any PEFT fields, the loader behavior is unchanged.

    The apply+load step is fail-loud: a missing or corrupt sidecar
    raises ``FileNotFoundError`` / ``ValueError`` /
    ``RuntimeError`` so the serving process refuses to start with a
    partial config — better than silently serving the un-adapted base
    model.
    """
    if not config.model_path:
        return _create_dummy_model_and_tokenizer(config)

    checkpoint = load_training_checkpoint(config.model_path)
    if checkpoint.model_obj is not None:
        # Quantized-model blob produced by ``llm-quantize``: the module is
        # self-contained (packed weights, scales, per-layer quantization
        # params), so skip ModelFactory reconstruction entirely.
        model = checkpoint.model_obj
        if not isinstance(model, DecoderModel):
            raise TypeError(f"quantized model blob must be a DecoderModel, got {type(model).__name__}")
    else:
        model = _build_decoder(
            serving_config=config,
            model_config=checkpoint.model_config,
            state_dict=checkpoint.model_state,
        )
    tokenizer = load_tokenizer(config)

    _apply_peft_if_configured(model, config)

    return model, tokenizer


def _apply_peft_if_configured(model: DecoderModel, config: ServingConfig) -> None:
    """Apply PEFT method (and optionally load + merge adapter) at serve time.

    Called by :func:`load_model_and_tokenizer` after the base model is
    built. Three cases:

    - ``peft_method`` + ``peft_adapter_path``: load the sidecar
      (auto-applies the method if needed).
    - ``peft_method`` only: just apply the method (useful for BitFit,
      where the adapter is the base weights and no separate sidecar
      is needed).
    - neither: no-op (loader behavior unchanged).

    When ``peft_merge`` is True and the method exposes a merge
    helper, the adapter is folded into the base weights to save
    per-token routing cost. The config validator rejects
    ``peft_merge=True`` for non-mergeable methods, so this branch
    always succeeds when reached.
    """
    if config.peft_method is None:
        return

    # Lazy import so the PEFT surface is only pulled in when actually
    # configured (the common CLI startup with no PEFT skips this).
    from llm.serving.peft_adapter import (
        load_peft_into_model,
        merge_peft_into_model,
    )

    if config.peft_adapter_path is not None:
        logger.info(
            "Loading PEFT adapter: method=%s path=%s",
            config.peft_method,
            config.peft_adapter_path,
        )
        load_peft_into_model(
            model,
            config.peft_method,
            config.peft_adapter_path,
            **config.peft_kwargs,
        )
    else:
        # Method-only path: apply the wrapper structure but no separate
        # sidecar to load. Useful for BitFit-only checkpoints where
        # the fine-tuned biases are already in ``model_state`` (saved
        # as part of the main training checkpoint).
        from llm.core.peft import apply_peft

        logger.info(
            "Applying PEFT method (no sidecar): method=%s kwargs=%s",
            config.peft_method,
            config.peft_kwargs,
        )
        apply_peft(model, config.peft_method, **config.peft_kwargs)

    if config.peft_merge:
        logger.info("Merging PEFT adapter into base weights: method=%s", config.peft_method)
        merge_peft_into_model(model, config.peft_method)


def _create_dummy_model_and_tokenizer(config: ServingConfig) -> tuple[DecoderModel, Any]:
    """Fallback demo model when no checkpoint is configured."""
    tokenizer = TokenizerFactory.from_printable_corpus()
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
    if not isinstance(model, DecoderModel):
        raise TypeError(f"dummy model must be a DecoderModel, got {type(model).__name__}")
    return model, tokenizer
