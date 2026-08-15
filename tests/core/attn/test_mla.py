from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

from llm.core.attn import MultiLatentAttention
from tests.support.devices import DEFAULT_DEVICE

if TYPE_CHECKING:
    from llm.core.paged_attention.paged_kv_cache import PagedKVCache

_DEVICE_STR = str(DEFAULT_DEVICE)


@pytest.fixture
def mla():
    return MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=16).to(DEFAULT_DEVICE)


@pytest.fixture
def input_tensor():
    return torch.randn(2, 10, 64, device=DEFAULT_DEVICE)  # [batch_size, seq_len, hidden_size]


@pytest.mark.slow
def test_mla_initialization(mla):
    """Test if MLA module is initialized correctly."""
    assert isinstance(mla, MultiLatentAttention)
    assert mla.hidden_size == 64
    assert mla.num_heads == 8
    assert mla.head_dim == 8  # 64 // 8
    torch.testing.assert_close(mla.scale, 1 / math.sqrt(8), atol=1e-12, rtol=1e-12)
    assert mla.dropout_p == 0.1
    assert not mla.is_causal
    assert mla.num_latents == 16
    assert mla.latent_dim == 64


@pytest.mark.slow
def test_mla_forward_shape(mla, input_tensor):
    """Test if forward pass maintains correct shape."""
    output = mla(input_tensor)
    assert output.shape == input_tensor.shape


@pytest.mark.slow
def test_mla_with_mask(mla, input_tensor):
    """Test MLA with attention mask."""
    # Create a simple attention mask
    attn_mask = torch.ones(2, 1, 1, 10, dtype=torch.bool, device=DEFAULT_DEVICE)  # [batch_size, 1, 1, seq_len]
    output = mla(input_tensor, attn_mask)
    assert output.shape == input_tensor.shape


@pytest.mark.slow
def test_mla_gradients(mla, input_tensor):
    """Test if gradients are computed correctly."""
    input_tensor.requires_grad_(True)
    output = mla(input_tensor)
    loss = output.sum()
    loss.backward()
    assert input_tensor.grad is not None
    assert not torch.isnan(input_tensor.grad).any()


@pytest.mark.slow
def test_mla_causal(mla, input_tensor):
    """Test causal MLA."""
    mla.is_causal = True
    output = mla(input_tensor)
    assert output.shape == input_tensor.shape


@pytest.mark.slow
def test_mla_different_num_latents():
    """Test MLA with different number of latents."""
    num_latents_list = [8, 16, 32]
    for num_latents in num_latents_list:
        mla = MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=num_latents).to(DEFAULT_DEVICE)
        input_tensor = torch.randn(2, 10, 64, device=DEFAULT_DEVICE)
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


@pytest.mark.slow
def test_mla_different_latent_dims():
    """Test MLA with different latent dimensions."""
    latent_dims = [32, 64, 128]
    for latent_dim in latent_dims:
        mla = MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=16, latent_dim=latent_dim).to(
            DEFAULT_DEVICE
        )
        # Use the same hidden_size for input tensor
        input_tensor = torch.randn(2, 10, 64, device=DEFAULT_DEVICE)  # hidden_size should match the model's hidden_size
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


@pytest.mark.slow
def test_mla_different_hidden_sizes():
    """Test MLA with different hidden sizes."""
    hidden_sizes = [32, 64, 128]
    for hidden_size in hidden_sizes:
        mla = MultiLatentAttention(hidden_size=hidden_size, num_heads=8, num_latents=16).to(DEFAULT_DEVICE)
        input_tensor = torch.randn(2, 10, hidden_size, device=DEFAULT_DEVICE)
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


@pytest.mark.slow
def test_mla_different_batch_sizes(mla):
    """Test MLA with different batch sizes."""
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 10, 64, device=DEFAULT_DEVICE)
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


@pytest.mark.slow
def test_mla_different_sequence_lengths(mla):
    """Test MLA with different sequence lengths."""
    seq_lengths = [5, 10, 20]
    for seq_len in seq_lengths:
        input_tensor = torch.randn(2, seq_len, 64, device=DEFAULT_DEVICE)
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


@pytest.mark.slow
def test_mla_norm_first():
    """Test MLA with norm_first=True."""
    mla = MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=16, norm_first=True).to(DEFAULT_DEVICE)
    input_tensor = torch.randn(2, 10, 64, device=DEFAULT_DEVICE)
    output = mla(input_tensor)
    assert output.shape == input_tensor.shape


def test_mla_plain_sublayer_without_norm_residual():
    """RIL ISS-139: with ``include_norm_residual=False`` (what the
    TransformerBlock passes for a plain sublayer), MLA must NOT own a norm
    or apply an internal residual — the block adds the residual. Before the
    fix MLA always built ``self.norm`` and applied an internal norm+residual,
    so the block's residual was DOUBLED and the already-normalized input was
    normalized AGAIN."""
    mla = MultiLatentAttention(
        hidden_size=32, num_heads=4, num_latents=8, include_norm_residual=False, norm_first=False
    ).to(DEFAULT_DEVICE)
    assert mla.norm is None, "plain sublayer must not own a norm"
    x = torch.randn(2, 8, 32, device=DEFAULT_DEVICE)
    out = mla(x)
    assert out.shape == x.shape


def test_mla_norm_owned_only_when_include_norm_residual():
    """When ``include_norm_residual=True`` MLA owns the norm (the previous
    behavior is preserved for direct users)."""
    mla = MultiLatentAttention(
        hidden_size=32, num_heads=4, num_latents=8, include_norm_residual=True, norm_first=True
    ).to(DEFAULT_DEVICE)
    assert mla.norm is not None


# --- KV cache forward path (T3 #31) --------------------------------------
#
# These tests verify the placeholder MLA + KV-cache contract:
#   - linear ``KVCache`` writes K/V from ``input_kv_proj`` and reads back
#     the cached context on decode steps;
#   - block-allocator ``PagedKVCache`` does the same for continuous
#     batching;
#   - incremental decode (prefill + step-by-step) is equivalent to a
#     full-sequence forward within the cache contract's tolerance.


def _make_mla_for_cache_test(seed: int = 0) -> MultiLatentAttention:
    """Tiny eval-mode MLA for KV-cache equivalence tests."""
    torch.manual_seed(seed)
    return (
        MultiLatentAttention(
            hidden_size=32,
            num_heads=4,
            num_latents=8,
            latent_dim=32,
            p=0.0,
            is_causal=False,
            include_norm_residual=False,
            bias=False,
        )
        .to(DEFAULT_DEVICE)
        .eval()
    )


@pytest.mark.slow
def test_mla_kv_cache_full_equivalence():
    """Incremental decode through the linear cache matches the full-seq forward.

    The placeholder MLA caches K, V from ``input_kv_proj``. Running
    prefill + N decode steps should produce the same per-position output
    as a single full-sequence forward (the cache only saves projection
    cost — it does not change the attention math).
    """
    from llm.core.kv_cache import KVCache

    torch.manual_seed(0)
    batch_size = 1
    seq_len = 6
    hidden_size = 32

    mla = _make_mla_for_cache_test(seed=0)

    x = torch.randn(batch_size, seq_len, hidden_size, device=DEFAULT_DEVICE)
    kv_cache = KVCache(
        max_batch_size=batch_size,
        max_seq_len=seq_len,
        num_kv_heads=mla.num_heads,
        head_dim=mla.head_dim,
        device=x.device,
        dtype=x.dtype,
    )

    # Reference: full-sequence forward with no cache.
    with torch.no_grad():
        ref_out = mla(x)

    # Incremental: prefill + decode one token at a time.
    with torch.no_grad():
        prefill = x[:, : seq_len - 1, :]
        decode_token = x[:, seq_len - 1 :, :]

        _ = mla(prefill, kv_cache=kv_cache)
        cached_out = mla(decode_token, kv_cache=kv_cache)

    # Last cached position must match the last reference position.
    torch.testing.assert_close(
        cached_out[:, 0, :],
        ref_out[:, -1, :],
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.slow
def test_mla_kv_cache_returns_full_output():
    """MLA + KV cache returns ``[B, S_new, H]`` (the new tokens' output)."""
    from llm.core.kv_cache import KVCache

    torch.manual_seed(0)
    batch_size = 2
    seq_len = 5
    hidden_size = 32

    mla = _make_mla_with_seed(7)
    x = torch.randn(batch_size, seq_len, hidden_size, device=DEFAULT_DEVICE)
    kv_cache = KVCache(
        max_batch_size=batch_size,
        max_seq_len=seq_len + 4,
        num_kv_heads=mla.num_heads,
        head_dim=mla.head_dim,
        device=x.device,
        dtype=x.dtype,
    )

    with torch.no_grad():
        out = mla(x, kv_cache=kv_cache)

    assert out.shape == (batch_size, seq_len, hidden_size)
    # The cache itself recorded the full prefill.
    assert kv_cache.seq_len == seq_len


@pytest.mark.slow
def test_mla_kv_cache_and_paged_kv_cache_mutually_exclusive():
    """Calling forward with both ``kv_cache`` and ``paged_kv_cache`` raises."""
    from llm.core.kv_cache import KVCache
    from llm.core.paged_attention.paged_kv_cache import PagedKVCache

    torch.manual_seed(0)
    hidden_size = 32
    num_heads = 4
    head_dim = hidden_size // num_heads

    mla = _make_mla_for_cache_test(seed=11)
    x = torch.randn(1, 1, hidden_size, device=DEFAULT_DEVICE)

    kv_cache = KVCache(
        max_batch_size=1,
        max_seq_len=8,
        num_kv_heads=num_heads,
        head_dim=head_dim,
    )
    paged = PagedKVCache(
        num_layers=1,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        num_blocks=4,
        block_size=4,
        device=_DEVICE_STR,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="Pass either kv_cache or paged_kv_cache"):
        mla(
            x,
            kv_cache=kv_cache,
            paged_kv_cache=paged,
            layer_idx=0,
            batch_indices=torch.tensor([0], dtype=torch.long),
        )


def _make_mla_with_seed(seed: int) -> MultiLatentAttention:
    """Tiny eval-mode MLA with explicit seed (per-test reproducibility)."""
    torch.manual_seed(seed)
    return (
        MultiLatentAttention(
            hidden_size=32,
            num_heads=4,
            num_latents=8,
            latent_dim=32,
            p=0.0,
            is_causal=False,
            include_norm_residual=False,
            bias=False,
        )
        .to(DEFAULT_DEVICE)
        .eval()
    )


# --- Paged KV cache forward path -----------------------------------------


def _build_paged_cache_for_test(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int = 8,
    block_size: int = 4,
) -> PagedKVCache:
    """Construct a small ``PagedKVCache`` for unit tests."""
    from llm.core.paged_attention.paged_kv_cache import PagedKVCache

    return PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        device=_DEVICE_STR,
        dtype=torch.float32,
    )


@pytest.mark.slow
def test_mla_paged_kv_cache_roundtrip_writes_and_reads():
    """Prefill through the paged cache writes blocks and returns the right shape."""
    torch.manual_seed(0)
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 4
    head_dim = hidden_size // num_heads
    batch_size = 1
    seq_len = 5
    seq_id = 0

    mla = _make_mla_for_cache_test(seed=13)
    paged = _build_paged_cache_for_test(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    x = torch.randn(batch_size, seq_len, hidden_size, device=DEFAULT_DEVICE)
    out = mla(
        x,
        paged_kv_cache=paged,
        layer_idx=0,
        batch_indices=torch.tensor([seq_id], dtype=torch.long),
    )

    assert out.shape == (batch_size, seq_len, hidden_size)
    assert paged.get_block_table(seq_id)  # non-empty
    assert paged.block_manager.get_num_tokens(seq_id) == seq_len


@pytest.mark.slow
def test_mla_paged_kv_cache_incremental_decode_equivalence():
    """Incremental decode through the paged cache matches full-seq forward."""
    torch.manual_seed(0)
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 4
    head_dim = hidden_size // num_heads
    batch_size = 1
    seq_len = 5
    seq_id = 0

    mla = _make_mla_for_cache_test(seed=17)
    paged = _build_paged_cache_for_test(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    x = torch.randn(batch_size, seq_len, hidden_size, device=DEFAULT_DEVICE)

    # Reference: full-sequence forward with no cache.
    with torch.no_grad():
        ref_out = mla(x)

    # Incremental: prefill + one decode step.
    with torch.no_grad():
        prefill = x[:, : seq_len - 1, :]
        decode_token = x[:, seq_len - 1 :, :]
        seq_ids = torch.tensor([seq_id], dtype=torch.long)

        _ = mla(prefill, paged_kv_cache=paged, layer_idx=0, batch_indices=seq_ids)
        cached_out = mla(
            decode_token,
            paged_kv_cache=paged,
            layer_idx=0,
            batch_indices=seq_ids,
        )

    torch.testing.assert_close(
        cached_out[:, 0, :],
        ref_out[:, -1, :],
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.slow
def test_mla_paged_kv_cache_requires_layer_idx_and_batch_indices():
    """The paged branch rejects callers that omit ``layer_idx`` or ``batch_indices``."""
    from llm.core.paged_attention.paged_kv_cache import PagedKVCache

    torch.manual_seed(0)
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 4
    head_dim = hidden_size // num_heads

    mla = _make_mla_for_cache_test(seed=19)
    paged = PagedKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=4,
        block_size=4,
        device=_DEVICE_STR,
        dtype=torch.float32,
    )
    x = torch.randn(1, 1, hidden_size, device=DEFAULT_DEVICE)

    with pytest.raises(ValueError, match="layer_idx is required"):
        mla(
            x,
            paged_kv_cache=paged,
            layer_idx=None,
            batch_indices=torch.tensor([0], dtype=torch.long),
        )
    with pytest.raises(ValueError, match="batch_indices is required"):
        mla(x, paged_kv_cache=paged, layer_idx=0, batch_indices=None)


@pytest.mark.slow
def test_mla_paged_kv_cache_decode_step_appends_block():
    """A single-token decode after a prefill extends the existing block table.

    Block allocation behaviour is owned by ``PagedKVCache.update``; this
    test just confirms MLA's call site wires through the standard
    per-row ``seq_id`` and reuses blocks correctly.
    """
    torch.manual_seed(0)
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 4
    head_dim = hidden_size // num_heads
    seq_id = 0

    mla = _make_mla_for_cache_test(seed=23)
    paged = _build_paged_cache_for_test(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=4,
    )

    # Prefill 5 tokens → 2 blocks.
    prefill = torch.randn(1, 5, hidden_size, device=DEFAULT_DEVICE)
    seq_ids = torch.tensor([seq_id], dtype=torch.long)
    mla(prefill, paged_kv_cache=paged, layer_idx=0, batch_indices=seq_ids)
    blocks_after_prefill = paged.get_block_table(seq_id)
    assert len(blocks_after_prefill) == 2

    # Decode one token → same block table, count = 6.
    decode = torch.randn(1, 1, hidden_size, device=DEFAULT_DEVICE)
    mla(decode, paged_kv_cache=paged, layer_idx=0, batch_indices=seq_ids)
    assert paged.get_block_table(seq_id) == blocks_after_prefill
    assert paged.block_manager.get_num_tokens(seq_id) == 6


def test_mla_latent_attention_heterogeneous_mask_uses_last_real_query_row():
    """Sweep HIGH: in a heterogeneous continuous-batching batch the engine's
    mask has q_len = the batch-max query length, with shorter rows (decode /
    shorter prefill) right-padded by all-masked rows. The old collapse
    ``attn_mask[:, :, -1:, :]`` took the LAST q_len row — a fully-masked pad
    row for those elements — so SDPA returned ZEROS for their latent
    attention (output depends only on bias: silent wrong tokens). The
    collapse must use each row's LAST REAL query row (the "current token"
    view the latents represent).
    """
    mla = MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=16).to(DEFAULT_DEVICE)
    mla.eval()

    batch, q_len, k_len = 2, 4, 8
    k = torch.randn(batch, 8, k_len, 8, device=DEFAULT_DEVICE)
    v = torch.randn(batch, 8, k_len, 8, device=DEFAULT_DEVICE)

    # Row 0: a 4-token prefill (real rows 0..3, causal). Row 1: a decode row
    # whose ONLY real query is at position 6 (row 0); rows 1..3 are padding
    # and stay fully masked (all-True).
    col = torch.arange(k_len, device=DEFAULT_DEVICE).view(1, 1, 1, -1)
    qpos = torch.arange(q_len, device=DEFAULT_DEVICE).view(1, 1, -1, 1)
    mask = torch.ones(batch, 1, q_len, k_len, dtype=torch.bool, device=DEFAULT_DEVICE)
    mask[0, 0] = col > qpos  # causal for the prefill row
    mask[1, 0, 0, :] = col[0, 0, 0, :] > 6  # decode row's single real query

    out = mla._latent_attention(k, v, batch, mask, is_causal=False)
    assert out.shape == (batch, mla.num_latents, 64)
    # The decode row must NOT collapse to zeros.
    assert not torch.allclose(out[1], torch.zeros_like(out[1]), atol=1e-6), (
        "padded decode row's latent attention collapsed to zeros (wrong mask row)"
    )
    # Heterogeneous batch == isolated batch for the affected row.
    out_iso = mla._latent_attention(k[1:2], v[1:2], 1, mask[1:2], is_causal=False)
    assert torch.allclose(out[1], out_iso[0], atol=1e-5), (
        "heterogeneous-batch result must match the isolated-row result"
    )


def test_mla_latent_attention_mask_length_equals_num_latents_is_still_consistent():
    """Sweep HIGH (== branch): when q_len happens to equal num_latents the old
    code used the mask UN-collapsed, mapping each static latent query to a
    DIFFERENT q_len row (the i-th latent attended over row i instead of the
    shared 'current token' mask). The latents all share one key-visibility
    mask regardless of ``num_latents``, so the full-mask result must equal
    the pre-collapsed single-row-mask reference (note: outputs differ per
    latent because each latent has its own learned query — only the key
    visibility is shared).
    """
    mla = MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=6).to(DEFAULT_DEVICE)
    mla.eval()
    q_len = k_len = 6
    batch = 1
    k = torch.randn(batch, 8, k_len, 8, device=DEFAULT_DEVICE)
    v = torch.randn(batch, 8, k_len, 8, device=DEFAULT_DEVICE)
    col = torch.arange(k_len, device=DEFAULT_DEVICE).view(1, 1, 1, -1)
    qpos = torch.arange(q_len, device=DEFAULT_DEVICE).view(1, 1, -1, 1)
    mask = (col > qpos).to(DEFAULT_DEVICE).bool().expand(batch, 1, q_len, k_len)

    out = mla._latent_attention(k, v, batch, mask, is_causal=False)
    # Reference: the canonical behavior is the last row's mask applied to all
    # num_latents latents (the pre-collapsed [B,1,1,k_len] form).
    ref = mla._latent_attention(k, v, batch, mask[:, :, -1:, :], is_causal=False)
    assert torch.allclose(out, ref, atol=1e-5), (
        "full-mask result differs from the shared-single-row-mask reference "
        "(latents were mapped to distinct q_len rows)"
    )


def test_mla_paged_kv_write_filters_right_padded_tokens():
    """Sweep MEDIUM: ``_paged_kv_write`` must append only each sequence's
    REAL tokens into the paged cache. The engine's batch is right-padded to
    the batch-max query length, so writing the whole padded slice per row
    appends pad-token K/V — each pad consumes a KV block slot, inflates
    ``get_num_tokens``, and can swallow the whole block pool for short-lived
    sequences. Mirror MHA's per-row real-length filtering (the causal mask's
    column-0 visibility run).
    """
    from llm.core.paged_attention.paged_kv_cache import PagedKVCache

    mla = MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=16).to(DEFAULT_DEVICE)
    mla.eval()
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=8,
        head_dim=8,
        num_blocks=16,
        block_size=4,
        device=_DEVICE_STR,
        dtype=torch.float32,
    )
    batch, q_len, k_len = 2, 4, 8
    k = torch.randn(batch, 8, q_len, 8, device=DEFAULT_DEVICE)
    v = torch.randn(batch, 8, q_len, 8, device=DEFAULT_DEVICE)
    batch_indices = torch.tensor([0, 1], device=DEFAULT_DEVICE)

    col = torch.arange(k_len, device=DEFAULT_DEVICE).view(1, 1, 1, -1)
    qpos = torch.arange(q_len, device=DEFAULT_DEVICE).view(1, 1, -1, 1)
    mask = torch.ones(batch, 1, q_len, k_len, dtype=torch.bool, device=DEFAULT_DEVICE)
    mask[0, 0] = col > qpos  # prefill row: 4 real query rows
    mask[1, 0, 0, :] = col[0, 0, 0, :] > 6  # decode row: 1 real query row
    lengths = (~mask[:, 0, :, 0]).sum(dim=-1)  # [4, 1]

    mla._paged_kv_write(k, v, cache, batch_indices, layer_idx=0, target_seq_len=k_len, lengths=lengths)
    # Only the real tokens may be stored per sequence.
    assert cache.block_manager.get_num_tokens(0) == int(lengths[0]), (
        f"prefill row stored {cache.block_manager.get_num_tokens(0)} tokens, expected {lengths[0]}"
    )
    assert cache.block_manager.get_num_tokens(1) == int(lengths[1]), (
        f"decode row stored {cache.block_manager.get_num_tokens(1)} tokens, expected {lengths[1]}"
    )
