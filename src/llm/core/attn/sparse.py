"""Unified sparse / streaming attention-mask dispatcher (TASK-242).

Collects the four attention-mask builders shipped in ROADMAP 15.1 / 15.2
(block-sparse, StreamingLLM, Longformer, BigBird) behind one public entry point
so a caller can pick a scheme by name and get the same boolean mask (plus the
shared ``mask_to_additive`` / ``coverage_fraction`` helpers).
"""

from __future__ import annotations

import torch

from llm.core.attn.big_bird import build_bigbird_mask
from llm.core.attn.block_sparse import build_block_sparse_mask, coverage_fraction, mask_to_additive
from llm.core.attn.longformer import build_longformer_mask
from llm.core.attn.streaming_llm import build_streamingllm_mask

__all__ = [
    "SUPPORTED_KINDS",
    "build_config_attention_mask",
    "build_sparse_attention_mask",
    "coverage_fraction",
    "mask_to_additive",
]

_SPARSE_BUILDERS = {
    "block_sparse": build_block_sparse_mask,
    "streaming": build_streamingllm_mask,
    "longformer": build_longformer_mask,
    "bigbird": build_bigbird_mask,
}

SUPPORTED_KINDS: tuple[str, ...] = tuple(_SPARSE_BUILDERS)


def build_sparse_attention_mask(kind: str, seq_len: int, *, causal: bool = True, **kwargs) -> torch.Tensor:
    """Return a boolean ``[seq_len, seq_len]`` attention mask for ``kind``.

    ``kind`` selects the scheme; ``causal`` and any scheme-specific keyword
    arguments are forwarded to the matching builder. An unknown ``kind`` fails
    with a helpful message listing the supported schemes.
    """
    builder = _SPARSE_BUILDERS.get(kind)
    if builder is None:
        raise ValueError(f"unknown sparse attention kind {kind!r}; expected one of {SUPPORTED_KINDS}")
    return builder(seq_len, causal=causal, **kwargs)


def build_config_attention_mask(config: object, seq_len: int) -> torch.Tensor | None:
    """Return the SDPA mask-out boolean mask for ``config.attn_sparse``.

    ``config`` is any object exposing an ``attn_sparse: dict | None`` (e.g.
    :class:`llm.training.core.config.ModelConfig`). When unset, returns ``None``
    (no sparse mask). When set, builds the scheme's *pattern-only* mask
    (``causal=False``) and returns the ``True``=mask-out boolean form expected by
    the repo's SDPA path, ready to be passed to a model forward's ``attn_mask``.

    Causality is intentionally delegated to the consuming model: a decoder
    forward applies its own causal masking (``is_causal``), so building a causal
    mask here would double-mask and break the full-coverage sparse == dense
    parity invariant (RIL TASK-243).
    """
    spar = getattr(config, "attn_sparse", None)
    if not spar:
        return None
    params = dict(spar)
    kind = params.pop("kind")
    params.pop("causal", None)  # causality is imposed by the consuming model
    allow = build_sparse_attention_mask(kind, seq_len, causal=False, **params)  # True = attend
    return (~allow).bool()  # True = mask out (SDPA convention)
