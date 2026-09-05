"""GGUF export backend: model state dict to a GGUF file.

This is the ``EXPORT_REGISTRY``-compatible layer: it takes a model and
writes a GGUF v3 file containing every ``state_dict()`` tensor. v1
policy:

- default type is F16 (``quantize=None``);
- ``quantize="q4_0"`` / ``"q8_0"`` block-quantizes tensors with at
  least two dimensions whose last dimension is a multiple of 32 and
  keeps the remaining tensors F16;
- ``quantize="q2_k"`` .. ``"q6_k"`` block-quantizes tensors (256-wide
  K-quant super-blocks) whose last dimension is a multiple of 256,
  keeping the remaining tensors F16 (rounds 147-151);
- when a requested block-quant policy silently falls back to F16 for a
  rank-eligible tensor (last dim not a multiple of 32/256 — a real
  footgun, e.g. ``hidden_size``/vocab not multiple of 32), a summary
  warning is emitted; 1-D biases staying F16 is convention and not
  warned (RIL TASK-316/ISS-353);
- metadata carries the standard ``general.*`` keys; user metadata wins
  over defaults.

Non-floating tensors are rejected explicitly (v1 scope). The format
layer (reader/writer/quant) is torch-free; only this module imports
torch, via the model argument.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn

from llm.export.gguf.spec import (
    EXPORT_TENSOR_TYPES,
    GGMLQuantizationType,
    GGUFError,
    can_quantize_k_shape,
    can_quantize_shape,
)
from llm.export.gguf.writer import GGUFWriter

logger = logging.getLogger(__name__)

_QUANT_NAME_TO_TYPE = {
    "f32": GGMLQuantizationType.F32,
    "f16": GGMLQuantizationType.F16,
    "q4_0": GGMLQuantizationType.Q4_0,
    "q2_k": GGMLQuantizationType.Q2_K,
    "q3_k": GGMLQuantizationType.Q3_K,
    "q8_0": GGMLQuantizationType.Q8_0,
    "q4_k": GGMLQuantizationType.Q4_K,
    "q5_k": GGMLQuantizationType.Q5_K,
    "q6_k": GGMLQuantizationType.Q6_K,
}

# Block-quant policies whose per-tensor fallback to F16 is worth surfacing.
_BLOCK_QUANT_TYPES = frozenset(
    {
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q2_K,
        GGMLQuantizationType.Q3_K,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
    }
)

# llama.cpp ``llama_ftype`` values used in ``general.file_type``.
_FILE_TYPE = {
    GGMLQuantizationType.F32: 0,  # ALL_F32
    GGMLQuantizationType.F16: 1,  # MOSTLY_F16
    GGMLQuantizationType.Q4_0: 2,  # MOSTLY_Q4_0
    GGMLQuantizationType.Q2_K: 10,  # MOSTLY_Q2_K
    GGMLQuantizationType.Q3_K: 11,  # MOSTLY_Q3_K
    GGMLQuantizationType.Q4_K: 3,  # MOSTLY_Q4_K
    GGMLQuantizationType.Q5_K: 6,  # MOSTLY_Q5_K
    GGMLQuantizationType.Q8_0: 7,  # MOSTLY_Q8_0
    GGMLQuantizationType.Q6_K: 13,  # MOSTLY_Q6_K
}


def _resolve_quant_type(quantize: str | GGMLQuantizationType | None) -> GGMLQuantizationType:
    if quantize is None:
        return GGMLQuantizationType.F16
    if isinstance(quantize, GGMLQuantizationType):
        ttype = quantize
    else:
        key = str(quantize).lower()
        if key not in _QUANT_NAME_TO_TYPE:
            raise ValueError(
                f"quantize must be one of {sorted(_QUANT_NAME_TO_TYPE)} or a GGMLQuantizationType, got {quantize!r}"
            )
        ttype = _QUANT_NAME_TO_TYPE[key]
    # The reader understands more types than the writer can produce (legacy
    # Q4_1/Q5_0/Q5_1 and IQ*). Refuse those reader-only types — they have no
    # ``general.file_type`` mapping — rather than crashing with a KeyError
    # later (round-75 review HIGH). The K-quant family is exportable and stays
    # in ``EXPORT_TENSOR_TYPES`` (rounds 147-151).
    if ttype not in EXPORT_TENSOR_TYPES:
        raise GGUFError(
            f"{ttype.name} is reader-supported but not exportable; "
            f"export supports {sorted(t.name for t in EXPORT_TENSOR_TYPES)}"
        )
    return ttype


def _default_metadata(
    model: nn.Module,
    model_name: str | None,
    quant_type: GGMLQuantizationType,
    user_metadata: dict[str, Any] | None,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "general.architecture": "llm",
        "general.name": model_name or type(model).__name__,
        "general.file_type": _FILE_TYPE[quant_type],
        "general.quantization_version": 2,
    }
    if model_config is not None:
        # Full architecture config as a JSON string so ``load_gguf_model`` can
        # rebuild the exact model (GGUF metadata is typed but has no nested
        # dict type; a JSON string round-trips losslessly). Keys are sorted for
        # deterministic output. (RIL round-71 GGUF loader milestone.)
        defaults["general.llm_model_config"] = json.dumps(model_config, sort_keys=True)
        # Persist the model's LIVE norm epsilon as its own key: ``ModelConfig``
        # has no ``norm_eps`` field, so a non-default eps (e.g. Qwen2's 1e-6)
        # cannot travel inside the config blob and a self-export would
        # round-trip to the loader default 1e-5 — RMS norm scaled differently
        # means wrong inference (RIL ISS-241; mirrors the hf_publisher fix in
        # CHG-201, which persists ``rms_norm_eps`` the same way).
        defaults["general.llm_norm_eps"] = float(_model_norm_eps(model))
    if user_metadata:
        defaults.update(user_metadata)
    return defaults


def _model_norm_eps(model: nn.Module) -> float:
    """The model's ACTUAL pre-norm epsilon (LayerNorm/RMSNorm ``eps``).

    ``DecoderModel`` holds it on its block pre-norms (``norm1``); fall back
    to the default (1e-5) when the architecture exposes no norm module.
    """
    blocks = getattr(model, "transformer_blocks", None)
    if blocks:
        for attr in ("norm1", "norm"):
            norm = getattr(blocks[0], attr, None)
            eps = getattr(norm, "eps", None)
            if eps is not None:
                return float(eps)
    return 1e-5


def _pick_tensor_type(
    arr: np.ndarray,
    quant_type: GGMLQuantizationType,
    quantize_min_ndim: int,
) -> GGMLQuantizationType:
    """Choose the on-disk type for one tensor under the export policy."""
    if quant_type in (GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q8_0):
        if arr.ndim >= quantize_min_ndim and can_quantize_shape(arr.shape):
            return quant_type
        return GGMLQuantizationType.F16
    if quant_type in (
        GGMLQuantizationType.Q2_K,
        GGMLQuantizationType.Q3_K,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
    ):
        if arr.ndim >= quantize_min_ndim and can_quantize_k_shape(arr.shape):
            return quant_type
        return GGMLQuantizationType.F16
    return quant_type


def export_to_gguf(
    model: nn.Module,
    output_path: str | Path,
    *,
    quantize: str | GGMLQuantizationType | None = None,
    metadata: dict[str, Any] | None = None,
    model_name: str | None = None,
    quantize_min_ndim: int = 2,
    model_config: dict[str, Any] | None = None,
) -> Path:
    """Export ``model.state_dict()`` to a GGUF v3 file.

    Args:
        model: The model to export (evaluated state, tensors are
            detached on CPU).
        output_path: Destination ``.gguf`` path; parent directories are
            created automatically.
        quantize: ``None`` (default) or ``"f16"`` writes F16 tensors;
            ``"f32"`` writes F32; ``"q4_0"`` / ``"q8_0"``
            block-quantizes eligible weight tensors (ndim >=
            ``quantize_min_ndim`` and last dim a multiple of 32) and
            keeps everything else F16.
        metadata: Extra ``general.*``-style metadata; overrides the
            built-in defaults (``general.name``, ``general.file_type``,
            ...).
        model_name: Override for ``general.name`` (defaults to the model
            class name).
        quantize_min_ndim: Minimum tensor rank eligible for
            block-quantization.
        model_config: Optional architecture config as a JSON-safe dict
            (e.g. ``ModelConfig.model_dump()``). When present it is
            persisted as ``general.llm_model_config`` so
            :func:`llm.export.gguf.loader.load_gguf_model` can rebuild
            the exact model — closing the export-only loop (round 71).

    Returns:
        The resolved output path.

    Raises:
        NotImplementedError: If the model has a non-floating tensor in
            its state dict (v1 scope).
        ValueError: For unknown ``quantize`` values.
    """
    quant_type = _resolve_quant_type(quantize)

    writer = GGUFWriter(output_path)
    for key, value in _default_metadata(model, model_name, quant_type, metadata, model_config).items():
        writer.add_metadata(key, value)

    degraded: list[tuple[str, tuple[int, ...]]] = []

    for name, tensor in model.state_dict().items():
        if not tensor.is_floating_point():
            raise NotImplementedError(
                f"GGUF exporter v1 only supports floating-point tensors; {name!r} has dtype {tensor.dtype}"
            )
        arr = tensor.detach().float().cpu().numpy()
        ttype = _pick_tensor_type(arr, quant_type, quantize_min_ndim)
        # A requested block-quant policy silently falling back to F16 for a
        # rank-eligible tensor (last dim not a multiple of 32/256) is a real
        # footgun — the export looks Q4_0 while a weight stays F16. Surface it
        # as ONE summary warning; 1-D biases staying F16 is convention and not
        # flagged (RIL TASK-316/ISS-353).
        if ttype != quant_type and quant_type in _BLOCK_QUANT_TYPES and arr.ndim >= quantize_min_ndim:
            degraded.append((name, tuple(arr.shape)))
        writer.add_tensor(name, arr, ggml_type=ttype)

    if degraded:
        samples = ", ".join(f"{n}{s}" for n, s in degraded[:8])
        logger.warning(
            "GGUF export: %d tensor(s)%s kept as F16 — shape not block-quantizable "
            "under requested %s, so per-tensor precision differs from general.file_type: %s",
            len(degraded),
            " (showing first 8)" if len(degraded) > 8 else "",
            quant_type.name,
            samples,
        )

    return writer.write()


def build_gguf_exporter(
    model: nn.Module,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Factory for the GGUF export target (``EXPORT_REGISTRY`` contract)."""
    return export_to_gguf(model, output_path, **kwargs)


__all__ = ["build_gguf_exporter", "export_to_gguf"]
