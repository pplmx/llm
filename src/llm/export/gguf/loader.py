"""High-level GGUF -> model loader (self-export load-back + llama.cpp import).

The format layer (reader/writer/quant/metadata) stays torch-free; this
module is deliberately high-level: it parses a GGUF file with
:class:`GGUFReader` and rebuilds an ``llm`` model from it.

Two file kinds are supported:

* **Self-exports** — files written by :func:`export_to_gguf` with
  ``model_config=`` carry the full architecture config as
  ``general.llm_model_config``; the exact model is rebuilt from it (round 71).
* **Foreign llama.cpp files** — a GGUF with no config blob but standard
  llama.cpp metadata (``general.architecture`` + ``{arch}.*`` keys) and
  llama-style tensor names (``token_embd`` / ``blk.N.attn_*`` /
  ``blk.N.ffn_*`` / ``output_norm`` / ``output``) is imported (round 72/73):
  the metadata rebuilds a :class:`ModelConfig` and
  :func:`llm.compat.weight_mapping.convert_gguf_weights` maps the tensor
  names into our state-dict naming (same q/k/v fusion and tied-head
  fallback as ``from_pretrained``).

Typical round trip::

    from llm.export.gguf import export_to_gguf, load_gguf_model

    export_to_gguf(model, "m.gguf", quantize="f16", model_config=cfg.model_dump())
    restored = load_gguf_model("m.gguf")

Import a third-party llama.cpp file::

    imported = load_gguf_model("llama-2-7b.Q8_0.gguf")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

from llm.compat.weight_mapping import convert_gguf_weights, convert_hf_to_combined_qkv
from llm.export.gguf.reader import GGUFReader
from llm.export.gguf.spec import GGUFError
from llm.models.decoder import DecoderModel
from llm.runtime.model_factory import ModelFactory
from llm.training.core.config import ModelConfig

logger = logging.getLogger(__name__)

__all__ = ["load_gguf_model"]


@runtime_checkable
class _SizedAttention(Protocol):
    """Structural type for attention backends exposing the MHA config dims."""

    num_heads: int
    num_kv_heads: int
    head_dim: int


# llama.cpp architecture names (gguf-py ``MODEL_ARCH_NAMES`` table) that map
# onto our DENSE Llama-style architecture — identical tensor layout
# (token_embd / blk.{i}.attn_* / blk.{i}.ffn_* / output_norm / output) and
# identical ``{arch}.*`` metadata keys. Everything else (mixtral, qwen2moe,
# qwen3moe, gemma, phi3, ...) is refused loudly rather than importing a model
# every weight of which lands at random init (RIL ISS-220 philosophy).
#
# ``llama2``/``llama3`` are defensive aliases: modern gguf writes the plain
# ``llama`` arch for all of Llama 1/2/3, but old converters declared the
# per-generation names. ``llama.mistral`` is the dense Mistral; ``qwen3``
# (dense) shares Qwen2's Llama-style layout.
_GGUF_ARCH_TO_MODEL_TYPE = {
    "llama": "llama",
    "llama2": "llama",
    "llama3": "llama",
    "mistral": "mistral",
    "qwen2": "qwen2",
    "qwen3": "qwen2",
}


def load_gguf_model(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Rebuild a model from a GGUF file — self-export or llama.cpp import.

    Args:
        path: GGUF file path.
        device: Optional target device (default: CPU).

    Returns:
        The rebuilt model in ``eval()`` mode with the exported/imported
        weights.

    Raises:
        GGUFError: If the file is malformed; if a self-export carries an
            invalid ``general.llm_model_config``; or if a foreign file is not
            a supported dense Llama-style architecture, is missing required
            ``{arch}.*`` metadata, carries tensors that do not map into
            ``llm`` state-dict naming, or uses an unsupported feature (RoPE
            scaling, non-standard head dims).
        RuntimeError: If the tensor names/shapes in a self-export do not match
            the model rebuilt from the embedded config (strict
            ``load_state_dict``).

    Note:
        F32/F16 files round-trip or import exactly (widened to float32);
        *block-quantized* files (Q4_0/Q8_0) come back dequantized and
        therefore approximately, within the quantizer's expected error.
    """
    reader = GGUFReader(path)

    raw_config = reader.metadata.get("general.llm_model_config")
    if isinstance(raw_config, str):
        return _load_self_export(reader, raw_config, device=device)
    return _load_foreign_gguf(reader, device=device)


def _load_self_export(
    reader: GGUFReader,
    raw_config: str,
    *,
    device: torch.device | str | None,
) -> nn.Module:
    """Load a GGUF this exporter wrote with ``model_config=`` (exact roundtrip)."""
    try:
        cfg = ModelConfig.model_validate(json.loads(raw_config))
    except Exception as exc:
        raise GGUFError(f"{reader.path}: invalid 'general.llm_model_config' JSON: {exc}") from exc

    # ``norm_eps`` is not a ModelConfig field (pydantic would silently drop
    # it), and the exporter persists the model's LIVE eps separately as
    # ``general.llm_norm_eps`` (CHG-209 era / RIL ISS-241, mirroring the
    # foreign path's ``_rms_norm_eps`` override below). Thread it through the
    # factory override so a self-export with a non-default eps round-trips
    # faithfully instead of silently resetting to 1e-5.
    raw_eps = reader.metadata.get("general.llm_norm_eps")
    norm_eps = float(raw_eps) if isinstance(raw_eps, (int, float)) else 1e-5
    model = ModelFactory.from_config(cfg, norm_eps=norm_eps)

    state: dict[str, torch.Tensor] = {}
    for name in reader.tensors:
        # ``read_tensor`` returns a read-only view over the file bytes for
        # F32/F16 (``np.frombuffer``); torch.from_numpy on a non-writable array
        # emits a UB warning, so take a writable copy before conversion.
        state[name] = torch.from_numpy(np.ascontiguousarray(reader.read_tensor(name)).copy())

    model.load_state_dict(state, strict=True)

    if device is not None:
        model.to(device)
    model.eval()
    return model


def _load_foreign_gguf(
    reader: GGUFReader,
    *,
    device: torch.device | str | None,
) -> nn.Module:
    """Import a third-party llama.cpp GGUF (no ``general.llm_model_config``)."""
    try:
        cfg, prefix = _gguf_metadata_to_model_config(reader.metadata, reader.path)
        # ``norm_eps`` is not a ModelConfig field (pydantic would silently drop
        # it), so thread the file's RMSNorm epsilon through the factory
        # override. Dense Llama-style GGUFs are always RMSNorm; the file's
        # epsilon is architecture-defining (1e-5 for Llama/Mistral, 1e-6 for
        # Qwen2).
        model = ModelFactory.from_config(cfg, norm_eps=_rms_norm_eps(reader.metadata, prefix))
    except GGUFError:
        raise
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        # Badly-typed / contradictory metadata must surface as a GGUF error
        # naming the file, not a raw ValueError traceback (round-73 review).
        raise GGUFError(f"{reader.path}: invalid llama.cpp metadata: {exc}") from exc
    if not isinstance(model, DecoderModel):
        raise GGUFError(f"{reader.path}: imported a {type(model).__name__}, expected a dense DecoderModel")

    gguf_state: dict[str, torch.Tensor] = {
        name: torch.from_numpy(np.ascontiguousarray(reader.read_tensor(name)).copy()) for name in reader.tensors
    }

    converted, unmapped = convert_gguf_weights(gguf_state, cfg.num_layers)
    if unmapped:
        raise GGUFError(
            f"{reader.path}: {len(unmapped)} tensor(s) have no mapping into llm naming "
            f"(first: {sorted(unmapped)[:5]}); this file is not a pure dense "
            f"Llama-style GGUF or carries extra tensors"
        )

    attn0 = model.transformer_blocks[0].self_attn
    if not isinstance(attn0, _SizedAttention):
        raise GGUFError(
            f"{reader.path}: attention backend {type(attn0).__name__} must expose "
            "num_heads/num_kv_heads/head_dim to fuse q/k/v"
        )
    converted = convert_hf_to_combined_qkv(
        converted,
        num_layers=cfg.num_layers,
        num_heads=attn0.num_heads,
        num_kv_heads=attn0.num_kv_heads,
        head_dim=attn0.head_dim,
    )

    try:
        missing, unexpected = model.load_state_dict(converted, strict=False)
    except RuntimeError as exc:
        # A real-file vocab/head-size mismatch surfaces as a torch shape error;
        # wrap it so the caller gets a GGUF error with the file path.
        raise GGUFError(f"{reader.path}: weights do not match the metadata-described architecture: {exc}") from exc

    # Tied embeddings (llama.cpp omits ``output.weight`` when the head is tied
    # to the embeddings): copy the embedding tensor into the LM head, mirroring
    # ``from_pretrained``'s standard HF tying semantics (RIL ISS-143).
    if "lm_head.weight" in missing and "embedding_layer.token_embeddings.weight" in converted:
        embedding_weight = converted["embedding_layer.token_embeddings.weight"]
        if tuple(embedding_weight.shape) == tuple(model.lm_head.weight.shape):
            with torch.no_grad():
                model.lm_head.weight.copy_(embedding_weight)
            missing = [k for k in missing if k != "lm_head.weight"]
            logger.info("%s: tied head — copied input embeddings into lm_head.", reader.path)

    if missing:
        raise GGUFError(
            f"{reader.path}: model keys not present in the GGUF ({len(missing)}, "
            f"first: {sorted(missing)[:5]}); the file's tensors do not cover the "
            f"architecture described by its metadata"
        )
    if unexpected:
        raise GGUFError(
            f"{reader.path}: {len(unexpected)} GGUF tensor(s) map to keys the "
            f"model does not have (first: {sorted(unexpected)[:5]})"
        )

    if device is not None:
        model.to(device)
    model.eval()
    return model


def _gguf_key_prefix(metadata: dict[str, Any]) -> str:
    """The file's llama.cpp metadata key prefix.

    gguf-py writes every model key as ``{arch}.*`` where ``{arch}`` is the
    exact ``general.architecture`` value — a Mistral file carries
    ``mistral.embedding_length``, a Qwen2 ``qwen2.*``, a Qwen3 ``qwen3.*``, so
    the prefix must come from the FILE, not from the mapped ``model_type``
    (round-73 review HIGH). Dense Llama-family files virtually always declare
    the plain ``llama`` arch; a file declaring a llama-family alias (old
    converter's ``llama2``/``llama3``) while still writing ``llama.*`` keys
    falls back to the ``llama`` prefix.
    """
    arch = str(metadata.get("general.architecture", "")).lower()
    if not arch:
        return ""
    if f"{arch}.embedding_length" in metadata:
        return arch
    if "llama.embedding_length" in metadata:
        return "llama"
    return arch


def _rms_norm_eps(metadata: dict[str, Any], prefix: str) -> float:
    """The file's RMSNorm epsilon (1e-5 default — Llama/Mistral reference)."""
    return float(metadata.get(f"{prefix}.attention.layer_norm_rms_epsilon", 1e-5))


def _gguf_metadata_to_model_config(
    metadata: dict[str, Any],
    path: Path,
) -> tuple[ModelConfig, str]:
    """Translate llama.cpp GGUF metadata into a dense Llama-style ModelConfig.

    Returns ``(config, key_prefix)`` so the caller can read eps/rope keys with
    the same prefix.
    """
    raw_arch = metadata.get("general.architecture")
    if not isinstance(raw_arch, str):
        raise GGUFError(
            f"{path}: not a compatible GGUF — no 'general.llm_model_config' "
            "(self-export) and no 'general.architecture' metadata (llama.cpp import)"
        )

    arch = raw_arch.lower()
    model_type = _GGUF_ARCH_TO_MODEL_TYPE.get(arch)
    if model_type is None:
        raise GGUFError(
            f"{path}: unsupported GGUF architecture {raw_arch!r} (supported dense "
            f"Llama-style: {sorted(set(_GGUF_ARCH_TO_MODEL_TYPE.values()))})"
        )

    prefix = _gguf_key_prefix(metadata)
    key = lambda suffix: f"{prefix}.{suffix}"  # noqa: E731 - short local alias

    missing = [
        s for s in ("embedding_length", "block_count", "attention.head_count", "vocab_size") if key(s) not in metadata
    ]
    if missing:
        raise GGUFError(f"{path}: missing required llama.cpp metadata: {', '.join(key(s) for s in missing)}")

    hidden_size = int(metadata[key("embedding_length")])
    num_layers = int(metadata[key("block_count")])
    num_heads = int(metadata[key("attention.head_count")])
    num_kv_heads = int(metadata.get(key("attention.head_count_kv"), num_heads))
    vocab_size = int(metadata[key("vocab_size")])
    intermediate_size = int(metadata[key("feed_forward_length")]) if key("feed_forward_length") in metadata else None
    max_seq_len = int(metadata.get(key("context_length"), 4096))
    rope_theta = float(metadata.get(key("rope.freq_base"), 10000.0))

    # Guard against non-standard per-head dims (``llama.attention.key_length``
    # / ``value_length`` emitted for archs whose head dim is not hidden//heads).
    # Our MHA hardcodes head_dim = hidden_size // num_heads, so a mismatching
    # file would import with every attention weight mis-sized and be rejected
    # by load_state_dict anyway — refuse up front with the reason.
    if hidden_size % num_heads != 0:
        raise GGUFError(
            f"{path}: embedding_length {hidden_size} not divisible by head_count "
            f"{num_heads} — non-standard head dimension (key_length/value_length), "
            f"not representable by llm's MHA"
        )
    head_dim = hidden_size // num_heads
    for suffix in ("attention.key_length", "attention.value_length"):
        if key(suffix) in metadata and int(metadata[key(suffix)]) != head_dim:
            raise GGUFError(
                f"{path}: {key(suffix)} = {metadata[key(suffix)]} != head_dim {head_dim}; "
                f"non-standard head dimensions are not representable by llm's MHA"
            )
    if num_heads % num_kv_heads != 0:
        raise GGUFError(
            f"{path}: head_count {num_heads} not divisible by head_count_kv "
            f"{num_kv_heads} — llm's MHA requires num_heads % num_kv_heads == 0"
        )

    # RoPE scaling (YaRN / linear / ...): our RoPE has no scaled-frequency
    # support, so a scaled file would import with the base freq_base and be
    # positionally wrong beyond the unscaled context. Refuse loudly rather
    # than ship a silently-wrong model (RIL ISS-220 philosophy; round-73
    # review MEDIUM — warn-and-import is a silent failure path).
    scaling = metadata.get(key("rope.scaling.type"))
    if scaling not in (None, "none"):
        raise GGUFError(
            f"{path}: RoPE scaling type {scaling!r} is not supported by llm's RoPE; "
            f"refusing rather than importing a model that is positionally wrong "
            f"beyond the base {max_seq_len} context. Use an unscaled GGUF."
        )

    # Dense Llama-style GGUFs are always pre-LN SwiGLU + RMSNorm + RoPE and
    # bias-free (attention/MLP/head). These are the model-defining flags that
    # would otherwise silently rebuild as LayerNorm / gelu-GLU / biased MHA with
    # every such tensor dropped (RIL ISS-056/062/129).
    return (
        ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            max_seq_len=max_seq_len,
            use_glu=True,
            mlp_activation="silu",
            norm_impl="rms_norm",
            norm_first=True,
            qkv_bias=False,
            mlp_bias=False,
            lm_head_bias=False,
            use_rope=True,
            rope_theta=rope_theta,
            use_alibi=False,
            attn_impl="mha",
            mlp_impl="mlp",
            dropout=0.0,
        ),
        prefix,
    )
