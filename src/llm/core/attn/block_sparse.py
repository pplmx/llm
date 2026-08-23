"""Block Sparse Attention mask builder (ROADMAP 15.2 / TASK-237).

Block-sparse attention (BigBird, Longformer-style) lets a query attend only to a
selected set of *key blocks* instead of every key: a small window around the
query, a few *global* key blocks that are always attended, and (optionally) a
fixed set of *random* key blocks. This module builds the resulting per-position
boolean mask over the existing additive/bool attention-mask machinery so any
backend (torch SDPA, etc.) can apply it.

A CPU parity invariant is central: when the window + global blocks together
cover every key block, the block-sparse mask is identical to the full
(Causal or full-allowed) mask — so block sparsity is a *constraint* on top of
dense attention, not a different computation, and the two agree exactly in the
non-sparse limit.
"""

from __future__ import annotations

import math

import torch


def build_block_sparse_mask(
    seq_len: int,
    *,
    block_size: int,
    window_blocks: int,
    global_blocks: int,
    random_blocks: int = 0,
    seed: int = 0,
    causal: bool = True,
) -> torch.Tensor:
    """Return a ``[S_q, S_k]`` bool mask (``True`` = attend).

    ``seq_len`` positions are grouped into ``block_size``-sized blocks. A query
    attends to keys that are
    - in the same or neighbouring ``window_blocks`` (local window), or
    - in one of the first ``global_blocks`` key blocks (global), or
    - (optionally) in a fixed seeded set of ``random_blocks`` key blocks.
    ``causal=True`` additionally forbids attending to future keys (``k > q``).
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if window_blocks < 0 or global_blocks < 0 or random_blocks < 0:
        raise ValueError("window_blocks / global_blocks / random_blocks must be >= 0")
    n_blocks = math.ceil(seq_len / block_size)
    global_blocks = min(global_blocks, n_blocks)
    if random_blocks > n_blocks:
        raise ValueError(f"random_blocks ({random_blocks}) cannot exceed block count ({n_blocks})")

    row = torch.arange(seq_len).view(-1, 1)
    col = torch.arange(seq_len).view(1, -1)
    query_block = row // block_size
    key_block = col // block_size

    global_allow = (key_block < global_blocks) | (query_block < global_blocks)
    window_allow = (query_block - key_block).abs() <= window_blocks

    random_allow = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    if random_blocks > 0:
        gen = torch.Generator()
        gen.manual_seed(seed)
        chosen = torch.randperm(n_blocks, generator=gen)[:random_blocks]
        random_allow = torch.isin(key_block, chosen)

    allow = global_allow | window_allow | random_allow
    if causal:
        causal_allow = col <= row
        allow = allow & causal_allow
    return allow


def mask_to_additive(mask: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert a bool attention mask to an additive ``0 / -inf`` bias.

    ``True`` (attend) -> 0; ``False`` (block) -> ``-inf`` so softmax weight is 0.
    """
    if mask.dtype != torch.bool:
        raise ValueError(f"mask_to_additive expects a bool mask, got {mask.dtype}")
    return torch.where(mask, torch.tensor(0.0, dtype=dtype), torch.tensor(float("-inf"), dtype=dtype))


def coverage_fraction(mask: torch.Tensor) -> float:
    """Fraction of ``[S_q, S_k]`` positions the sparse pattern actually allows."""
    return float(mask.sum().item() / mask.numel())
