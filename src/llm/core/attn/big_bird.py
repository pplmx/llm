"""BigBird attention mask builder (ROADMAP 15.2 / TASK-241).

BigBird (Zaheer et al. 2020) combines three attention patterns: a handful of
**global** tokens, a **local window**, and a fixed set of **random** key blocks.
That pattern is exactly what the block-sparse mask builder already implements
(``global_blocks`` + ``window_blocks`` + seeded ``random_blocks``), so this
module surfaces it under an explicit BigBird API on the shared machinery, keeping
the same semantics and CPU parity invariants as block sparsity.
"""

from __future__ import annotations

import torch

from llm.core.attn.block_sparse import build_block_sparse_mask, coverage_fraction, mask_to_additive

__all__ = ["build_bigbird_mask", "coverage_fraction", "mask_to_additive"]


def build_bigbird_mask(
    seq_len: int,
    *,
    block_size: int,
    num_global_blocks: int,
    window_blocks: int,
    num_random_blocks: int = 0,
    seed: int = 0,
    causal: bool = True,
) -> torch.Tensor:
    """Return BigBird's ``[S_q, S_k]`` boolean attention mask (``True`` = attend).

    Delegates to :func:`llm.core.attn.block_sparse.build_block_sparse_mask` with
    BigBird's structural arguments (global + local window + seeded random blocks).
    """
    return build_block_sparse_mask(
        seq_len,
        block_size=block_size,
        window_blocks=window_blocks,
        global_blocks=num_global_blocks,
        random_blocks=num_random_blocks,
        seed=seed,
        causal=causal,
    )
