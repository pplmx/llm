"""StreamingLLM attention-sink mask builder (ROADMAP 15.1 / TASK-238).

StreamingLLM (Xiao et al. 2023) keeps a rolling-context LLM stable over long
streams by always attending to the *first few* positions (the **attention
sink**) together with the most recent **window** of positions, and never to the
old-but-not-sink interior tokens — those absent keys are what would otherwise
destabilize the streaming attention.

This module builds the resulting per-position boolean mask over the same
additive/bool attention-mask machinery as the block-sparse slice, so any
backend (torch SDPA, etc.) can apply it.

CPU parity invariant: when the sink covers every position (``num_sink >= seq_len``)
or the window covers every past position, the streaming mask is identical to the
dense causal mask — i.e. streaming attention is a *constraint* on dense attention
and agrees with it exactly in the non-sparse limit.
"""

from __future__ import annotations

import torch

from llm.core.attn.block_sparse import coverage_fraction, mask_to_additive

__all__ = ["build_streamingllm_mask", "coverage_fraction", "mask_to_additive"]


def build_streamingllm_mask(
    seq_len: int,
    *,
    num_sink: int,
    window_size: int,
    causal: bool = True,
) -> torch.Tensor:
    """Return a ``[S_q, S_k]`` bool mask (``True`` = attend).

    A query attends to keys that are either
    - one of the first ``num_sink`` positions (attention sink, always attended), or
    - within the trailing ``window_size`` positions ending at the query (recent).
    ``causal=True`` additionally forbids attending to future keys.
    """
    if num_sink < 0 or window_size < 0:
        raise ValueError(f"num_sink / window_size must be >= 0, got {num_sink} / {window_size}")

    row = torch.arange(seq_len).view(-1, 1)
    col = torch.arange(seq_len).view(1, -1)

    sink_allow = col < num_sink
    window_allow = (col >= (row - window_size)) & (col <= row)

    allow = sink_allow | window_allow
    if causal:
        allow = allow & (col <= row)
    else:
        # Non-causal streaming still keeps the sink and a symmetric recent window.
        near = (col >= row - window_size) & (col <= row + window_size)
        allow = sink_allow | near
    return allow
