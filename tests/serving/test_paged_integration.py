import pytest
import torch

from llm.core.paged_attention.attention import paged_attention_forward
from llm.core.paged_attention.paged_kv_cache import PagedKVCache
from tests.support.devices import DEFAULT_DEVICE

_DEVICE_STR = str(DEFAULT_DEVICE)


def test_end_to_end_paged_inference():
    """Test paged attention workflow with multiple sequences."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=16,
        num_blocks=32,
        block_size=16,
        device=_DEVICE_STR,
        dtype=torch.float32,
    )

    seq1_k = torch.randn(1, 10, 2, 16, device=DEFAULT_DEVICE)
    seq1_v = torch.randn(1, 10, 2, 16, device=DEFAULT_DEVICE)
    block_ids_1 = cache.update(seq_id=1, k_new=seq1_k, v_new=seq1_v)

    seq2_k = torch.randn(1, 8, 2, 16, device=DEFAULT_DEVICE)
    seq2_v = torch.randn(1, 8, 2, 16, device=DEFAULT_DEVICE)
    block_ids_2 = cache.update(seq_id=2, k_new=seq2_k, v_new=seq2_v)

    max_blocks = max(len(block_ids_1), len(block_ids_2))
    block_tables = torch.zeros(2, max_blocks, dtype=torch.long, device=DEFAULT_DEVICE)
    block_tables[0, : len(block_ids_1)] = torch.tensor(block_ids_1, device=DEFAULT_DEVICE)
    block_tables[1, : len(block_ids_2)] = torch.tensor(block_ids_2, device=DEFAULT_DEVICE)
    seq_lens = torch.tensor([10, 8], device=DEFAULT_DEVICE)

    q = torch.randn(2, 4, 1, 16, device=DEFAULT_DEVICE)
    output = paged_attention_forward(
        q=q,
        k_cache=cache.k_cache,
        v_cache=cache.v_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        num_kv_heads=2,
    )

    assert output.shape == (2, 4, 1, 16)

    cache.free(seq_id=1)
    cache.free(seq_id=2)


def test_decode_row_attends_to_full_context_when_mixed_with_prefill():
    """Regression (RIL ISS-048): a decode row (single query token) in a batch
    that also contains a prefill row must attend to its full cached context,
    not just key position 0.

    The kernel rebuilds its own mask from ``seq_lens`` and applies the
    causal overlay (``col_idx > row_idx``) to EVERY row whenever the batch
    max query length is > 1 (i.e. any prefill row is present). The engine
    left-pads decode rows so their real query sits at local index 0, where
    that overlay masks all keys beyond position 0 — decode output becomes
    attention over the first prompt token only.

    Construction: q = ones (uniform), k = zeros (equal dot products) with a
    single head, V = [1,0] at key 0 and [0,0] elsewhere. With N = seq_len
    visible keys, uniform softmax over N keys yields output [1/N, 0] (norm
    1/N). If only key 0 is visible the output is [1, 0] (norm 1).
    """
    device = DEFAULT_DEVICE
    num_kv_heads = 1
    head_dim = 2
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=32,
        block_size=16,
        device=str(device),
        dtype=torch.float32,
    )

    # Decode row: seq_len fully cached context tokens.
    seq_len = 8
    k_full = torch.zeros(1, seq_len, num_kv_heads, head_dim, device=device)
    v_full = torch.zeros(1, seq_len, num_kv_heads, head_dim, device=device)
    v_full[0, 0, 0, :] = torch.tensor([1.0, 0.0], device=device)  # distinctive value at key 0
    block_ids = cache.update(seq_id=1, k_new=k_full, v_new=v_full)

    block_tables = torch.zeros(1, len(block_ids), dtype=torch.long, device=device)
    block_tables[0, :] = torch.tensor(block_ids, device=device)
    seq_lens = torch.tensor([seq_len], device=device)

    # Single decode query row listening on the FULL context.
    q = torch.ones(1, 1, 1, head_dim, device=device)

    # 1) Pure-decode batch (_query_len == 1): the kernel skips the causal
    #    overlay today, so it should already attend to all N keys.
    out_pure = paged_attention_forward(
        q=q,
        k_cache=cache.k_cache,
        v_cache=cache.v_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        num_kv_heads=num_kv_heads,
    )
    expected_uniform = torch.tensor([[1.0 / seq_len, 0.0]], device=device)
    assert torch.allclose(out_pure[0, 0, 0], expected_uniform, atol=1e-5), (
        f"pure-decode should attend to all {seq_len} keys: got {out_pure[0, 0, 0]}"
    )

    # 2) Mixed batch: the same decode row alongside a prefill row whose max
    #    query length is > 1. The global ``_query_len > 1`` branch must NOT
    #    apply the causal overlay to the decode row.
    cache2 = PagedKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=32,
        block_size=16,
        device=str(device),
        dtype=torch.float32,
    )
    cache2.update(seq_id=1, k_new=k_full, v_new=v_full)  # decode row (identical)
    prefill_k = torch.zeros(1, 4, num_kv_heads, head_dim, device=device)
    prefill_v = torch.zeros(1, 4, num_kv_heads, head_dim, device=device)
    cache2.update(seq_id=2, k_new=prefill_k, v_new=prefill_v)

    block_tables2 = torch.full((2, 2), -1, dtype=torch.long, device=device)
    b1 = cache2.get_block_table(1)
    b2 = cache2.get_block_table(2)
    block_tables2[0, : len(b1)] = torch.tensor(b1, device=device)
    block_tables2[1, : len(b2)] = torch.tensor(b2, device=device)
    seq_lens2 = torch.tensor([seq_len, 4], device=device)

    # Mixed batch: row 0 = decode (1 query), row 1 = prefill (4 queries).
    q_mixed = torch.zeros(2, 1, 4, head_dim, device=device)
    q_mixed[0, 0, 0, :] = torch.ones(head_dim, device=device)  # real decode query at local idx 0
    q_mixed[1, 0, :, :] = torch.ones(4, head_dim, device=device)  # prefill queries

    out_mixed = paged_attention_forward(
        q=q_mixed,
        k_cache=cache2.k_cache,
        v_cache=cache2.v_cache,
        block_tables=block_tables2,
        seq_lens=seq_lens2,
        num_kv_heads=num_kv_heads,
        # Production path (MHA) passes per-row real query lengths so the
        # causal overlay only applies to prefill rows.
        query_lens=torch.tensor([1, 4], device=device),
    )
    decode_out = out_mixed[0, 0, 0]
    assert torch.allclose(decode_out, expected_uniform, atol=1e-5), (
        f"decode row in a mixed batch must still attend to all {seq_len} keys: "
        f"got {decode_out} (only-key-0 would give [1.0, 0.0])"
    )
    cache.free(seq_id=1)
    cache2.free(seq_id=1)
    cache2.free(seq_id=2)


@pytest.mark.quick
def test_prefix_cache_integration_flow():
    """Test full prefix cache integration flow."""
    dev = _DEVICE_STR
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=16,
        num_blocks=64,
        block_size=16,
        device=dev,
        enable_prefix_cache=True,
        max_prefixes=10,
    )

    prefix_tokens = [1, 2, 3, 4, 5]
    k_prefix = torch.randn(1, 5, 2, 16, device=DEFAULT_DEVICE)
    v_prefix = torch.randn(1, 5, 2, 16, device=DEFAULT_DEVICE)

    block_ids = cache.update(seq_id=1, k_new=k_prefix, v_new=v_prefix)
    cache.add_prefix(seq_id=1, prefix_tokens=prefix_tokens, block_ids=block_ids)

    cached_blocks = cache.try_get_prefix_blocks(prefix_tokens)
    assert cached_blocks == block_ids

    k_new = torch.randn(1, 3, 2, 16, device=DEFAULT_DEVICE)
    v_new = torch.randn(1, 3, 2, 16, device=DEFAULT_DEVICE)
    new_block_ids = cache.update(seq_id=2, k_new=k_new, v_new=v_new)

    assert cache.get_block_table(1) == block_ids
    assert cache.get_block_table(2) == new_block_ids
