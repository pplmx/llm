"""Paged Attention forward implementation."""

import torch
from torch import Tensor


def paged_attention_forward(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    seq_lens: Tensor,
    num_kv_heads: int,
    block_size: int = 16,
    query_lens: Tensor | None = None,
) -> Tensor:
    """Paged attention forward pass.

    Supports both prefill (``q`` with multiple query tokens) and decode
    (``q`` with a single query token per sequence). Each row of ``q``
    attends to its full cached context (the first ``seq_lens[b]`` tokens
    of sequence ``b``), regardless of how many query tokens that row
    carries. The original Python-fallback kernel gathered the whole k/v
    slice per sequence regardless of ``S_q``; the multi-token generalisation
    just lets the matmul produce ``S_q`` outputs instead of one.

    Args:
        q: Query tensor [batch, num_heads, query_len, head_dim].
        k_cache: KV cache tensor. Either the per-layer slice
            ``[num_blocks, num_kv_heads, block_size, head_dim]`` (the
            production path — caller slices ``PagedKVCache.k_cache[layer_idx]``
            before passing) or the full ``PagedKVCache`` shape
            ``[num_layers, num_blocks, num_kv_heads, block_size, head_dim]``
            (legacy / direct tests). When 5-D, the layer axis is collapsed
            by taking index 0; for a multi-layer model the caller must slice.
        v_cache: Same shape as ``k_cache``.
        block_tables: [batch, max_blocks] physical block IDs per sequence.
        seq_lens: Current sequence lengths [batch].
        num_kv_heads: Number of KV heads.
        block_size: Tokens per block.
        query_lens: Optional per-row number of REAL query tokens [batch].
            In continuous batching, ``q`` is left-padded to the batch-max
            query length so a decode row's single query lives at local index
            0, and the causal overlay (which is only meaningful for prefill
            rows with multiple query tokens) must not mask a decode row's
            full context (RIL ISS-048). When ``None`` (direct tests /
            deprecated callers), the overlay applies to every row whenever
            ``query_len > 1``, mirroring the historical behaviour.

    Returns:
        Attention output tensor [batch, num_heads, query_len, head_dim].
    """
    batch_size, num_heads, _query_len, head_dim = q.shape
    if k_cache.ndim == 5:
        # Legacy / direct-test path: full ``PagedKVCache`` buffer.
        # The caller should have sliced already for a multi-layer model;
        # we fall back to layer 0 to match the historical behaviour.
        k_cache = k_cache[0]
        v_cache = v_cache[0]
    num_blocks = k_cache.shape[0]

    max_seq_len = seq_lens.max().item()

    k_gathered = []
    v_gathered = []

    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        num_blocks_needed = (seq_len + block_size - 1) // block_size

        seq_block_ids = block_tables[b, :num_blocks_needed].tolist()

        k_seq = []
        v_seq = []
        for i, block_id in enumerate(seq_block_ids):
            if block_id < 0 or block_id >= num_blocks:
                continue
            start = 0
            end = seq_len - (num_blocks_needed - 1) * block_size if i == num_blocks_needed - 1 else block_size

            k_seq.append(k_cache[block_id, :, start:end, :])
            v_seq.append(v_cache[block_id, :, start:end, :])

        if k_seq:
            k_full = torch.cat(k_seq, dim=1)
            v_full = torch.cat(v_seq, dim=1)
        else:
            # No valid blocks for this sequence (e.g. seq_len == 0 or all
            # block IDs out of range): skip to the padding step below by
            # starting from an empty tensor that pad_len will extend.
            k_full = k_cache.new_zeros(num_kv_heads, 0, head_dim)
            v_full = v_cache.new_zeros(num_kv_heads, 0, head_dim)

        if k_full.shape[1] < max_seq_len:
            pad_len = int(max_seq_len - k_full.shape[1])
            k_full = torch.cat([k_full, k_full.new_zeros(num_kv_heads, pad_len, head_dim)], dim=1)
            v_full = torch.cat([v_full, v_full.new_zeros(num_kv_heads, pad_len, head_dim)], dim=1)

        k_gathered.append(k_full)
        v_gathered.append(v_full)

    k_full = torch.stack(k_gathered, dim=0).to(q.device)
    v_full = torch.stack(v_gathered, dim=0).to(q.device)

    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        k_full = k_full.repeat_interleave(repeat_factor, dim=1)
        v_full = v_full.repeat_interleave(repeat_factor, dim=1)

    scale = head_dim**-0.5
    attn_weights = torch.matmul(q, k_full.transpose(-2, -1)) * scale

    # ``k_full`` is padded with zeros to the batch-max context length, so
    # columns beyond each row's real ``seq_lens[b]`` must be masked or they
    # would participate in the softmax and distort the attention weights.
    col_idx = torch.arange(max_seq_len, device=q.device).reshape(1, 1, 1, -1)
    attn_mask = col_idx >= seq_lens.reshape(-1, 1, 1, 1)  # [B, 1, 1, max_seq_len]
    if _query_len > 1:
        # Causality between query positions (query row ``s`` attends to keys
        # ``0..s``) is only meaningful for prefill rows that carry multiple
        # real query tokens. In continuous batching ``q`` is left-padded to
        # the batch-max query length, so a decode row's single real query sits
        # at local index 0 with padding after it; applying the overlay to that
        # row masks every key > 0, collapsing decode attention onto key 0 and
        # producing garbage output (RIL ISS-048). Gate the overlay per-row on
        # having more than one real query token.
        if query_lens is not None:
            prefill_rows = query_lens.reshape(-1, 1, 1, 1) > 1  # [B, 1, 1, 1]
            row_idx = torch.arange(_query_len, device=q.device).reshape(1, 1, -1, 1)
            causal = col_idx > row_idx  # [1, 1, query_len, max_seq_len]
            attn_mask = attn_mask | (causal & prefill_rows)
        else:
            # Legacy / direct-test callers that don't pass query_lens: apply
            # the overlay to every row (historical behaviour).
            row_idx = torch.arange(_query_len, device=q.device).reshape(1, 1, -1, 1)
            attn_mask = attn_mask | (col_idx > row_idx)
    attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))

    attn_weights = torch.softmax(attn_weights, dim=-1)

    output = torch.matmul(attn_weights, v_full)

    return output
