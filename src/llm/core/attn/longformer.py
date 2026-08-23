"""Longformer-style dilated sliding-window attention mask (ROADMAP 15.2 / TASK-240).

Longformer (Beltagy et al. 2020) builds near-linear attention from a **dilated
sliding window** plus a few **global** tokens: within the window a query attends
to every ``dilation``-th key (reaching farther with the same budget), and the
first ``num_global`` positions are attended by every token.

This module builds the resulting per-position boolean mask over the same
additive/bool attention-mask machinery as the block-sparse and StreamingLLM
slices (reusing ``mask_to_additive`` / ``coverage_fraction``).

CPU parity invariant: with ``dilation=1`` and ``num_global=0`` the mask is
identical to the plain sliding-window mask — dilation is a *constraint* on top
of the window, not a different computation.
"""

from __future__ import annotations

import torch

from llm.core.attn.block_sparse import coverage_fraction, mask_to_additive

__all__ = ["build_longformer_mask", "coverage_fraction", "mask_to_additive"]


def build_longformer_mask(
    seq_len: int,
    *,
    window_size: int,
    dilation: int = 1,
    num_global: int = 0,
    causal: bool = True,
) -> torch.Tensor:
    """Return a ``[S_q, S_k]`` bool mask (``True`` = attend).

    A query attends to keys that are
    - one of the first ``num_global`` positions (global tokens), or
    - within ``window_size`` of the query **and** at a position the window
      actually samples (every ``dilation``-th step).
    ``causal=True`` additionally forbids attending to future keys.
    """
    if window_size < 0 or num_global < 0:
        raise ValueError(f"window_size / num_global must be >= 0, got {window_size} / {num_global}")
    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")

    row = torch.arange(seq_len).view(-1, 1)
    col = torch.arange(seq_len).view(1, -1)

    distance = (row - col).abs()
    within = distance <= window_size
    sampled = (distance % dilation) == 0
    allow = within & sampled

    global_allow = col < num_global
    allow = allow | global_allow

    if causal:
        allow = allow & (col <= row)
    return allow
