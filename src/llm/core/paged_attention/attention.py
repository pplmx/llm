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
) -> Tensor:
    """Paged attention forward pass.

    Args:
        q: Query tensor for current token(s)
        k_cache: KV cache with block-level storage
        v_cache: Same as k_cache
        block_tables: [batch, max_blocks] physical block IDs for each sequence
        seq_lens: Current sequence lengths
        num_kv_heads: Number of KV heads
        block_size: Tokens per block

    Returns:
        Attention output tensor [batch, num_heads, 1, head_dim]
    """
    batch_size, num_heads, _, head_dim = q.shape
    num_layers, num_blocks, _, _, _ = k_cache.shape

    k_cache = k_cache[0]
    v_cache = v_cache[0]

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

        k_full = torch.cat(k_seq, dim=1)
        v_full = torch.cat(v_seq, dim=1)

        if k_full.shape[1] < max_seq_len:
            pad_len = max_seq_len - k_full.shape[1]
            k_full = torch.cat([k_full, k_full.new_zeros(num_kv_heads, pad_len, head_dim)], dim=1)
            v_full = torch.cat([v_full, v_full.new_zeros(num_kv_heads, pad_len, head_dim)], dim=1)

        k_gathered.append(k_full)
        v_gathered.append(v_full)

    k_full = torch.stack(k_gathered, dim=0)
    v_full = torch.stack(v_gathered, dim=0)

    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        k_full = k_full.repeat_interleave(repeat_factor, dim=1)
        v_full = v_full.repeat_interleave(repeat_factor, dim=1)

    scale = head_dim**-0.5
    attn_weights = torch.matmul(q, k_full.transpose(-2, -1)) * scale

    attn_weights = torch.softmax(attn_weights, dim=-1)

    output = torch.matmul(attn_weights, v_full)

    return output
