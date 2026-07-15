import math

import pytest
import torch

from llm.core.attn import MultiLatentAttention


@pytest.fixture
def mla():
    return MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=16)


@pytest.fixture
def input_tensor():
    return torch.randn(2, 10, 64)  # [batch_size, seq_len, hidden_size]


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
    attn_mask = torch.ones(2, 1, 1, 10, dtype=torch.bool)  # [batch_size, 1, 1, seq_len]
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
        mla = MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=num_latents)
        input_tensor = torch.randn(2, 10, 64)
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


@pytest.mark.slow
def test_mla_different_latent_dims():
    """Test MLA with different latent dimensions."""
    latent_dims = [32, 64, 128]
    for latent_dim in latent_dims:
        mla = MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=16, latent_dim=latent_dim)
        # Use the same hidden_size for input tensor
        input_tensor = torch.randn(2, 10, 64)  # hidden_size should match the model's hidden_size
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


@pytest.mark.slow
def test_mla_different_hidden_sizes():
    """Test MLA with different hidden sizes."""
    hidden_sizes = [32, 64, 128]
    for hidden_size in hidden_sizes:
        mla = MultiLatentAttention(hidden_size=hidden_size, num_heads=8, num_latents=16)
        input_tensor = torch.randn(2, 10, hidden_size)
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


@pytest.mark.slow
def test_mla_different_batch_sizes(mla):
    """Test MLA with different batch sizes."""
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 10, 64)
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


@pytest.mark.slow
def test_mla_different_sequence_lengths(mla):
    """Test MLA with different sequence lengths."""
    seq_lengths = [5, 10, 20]
    for seq_len in seq_lengths:
        input_tensor = torch.randn(2, seq_len, 64)
        output = mla(input_tensor)
        assert output.shape == input_tensor.shape


@pytest.mark.slow
def test_mla_norm_first():
    """Test MLA with norm_first=True."""
    mla = MultiLatentAttention(hidden_size=64, num_heads=8, num_latents=16, norm_first=True)
    input_tensor = torch.randn(2, 10, 64)
    output = mla(input_tensor)
    assert output.shape == input_tensor.shape


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
    return MultiLatentAttention(
        hidden_size=32,
        num_heads=4,
        num_latents=8,
        latent_dim=32,
        p=0.0,
        is_causal=False,
        include_norm_residual=False,
        bias=False,
    ).eval()


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
    num_layers = 1

    mla = _make_mla_for_cache_test(seed=0)

    x = torch.randn(batch_size, seq_len, hidden_size)
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
        prefill = x[:, :seq_len - 1, :]
        decode_token = x[:, seq_len - 1:, :]

        _ = mla(prefill, kv_cache=kv_cache)
        cached_out = mla(decode_token, kv_cache=kv_cache)

    # Last cached position must match the last reference position.
    torch.testing.assert_close(
        cached_out[:, 0, :], ref_out[:, -1, :], atol=1e-5, rtol=1e-5,
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
    x = torch.randn(batch_size, seq_len, hidden_size)
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
    x = torch.randn(1, 1, hidden_size)

    kv_cache = KVCache(
        max_batch_size=1, max_seq_len=8,
        num_kv_heads=num_heads, head_dim=head_dim,
    )
    paged = PagedKVCache(
        num_layers=1, num_kv_heads=num_heads, head_dim=head_dim,
        num_blocks=4, block_size=4,
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
    return MultiLatentAttention(
        hidden_size=32,
        num_heads=4,
        num_latents=8,
        latent_dim=32,
        p=0.0,
        is_causal=False,
        include_norm_residual=False,
        bias=False,
    ).eval()


# --- Paged KV cache forward path -----------------------------------------


def _build_paged_cache_for_test(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int = 8,
    block_size: int = 4,
) -> "PagedKVCache":
    """Construct a small ``PagedKVCache`` for unit tests."""
    from llm.core.paged_attention.paged_kv_cache import PagedKVCache

    return PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        device="cpu",
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
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )

    x = torch.randn(batch_size, seq_len, hidden_size)
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
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )

    x = torch.randn(batch_size, seq_len, hidden_size)

    # Reference: full-sequence forward with no cache.
    with torch.no_grad():
        ref_out = mla(x)

    # Incremental: prefill + one decode step.
    with torch.no_grad():
        prefill = x[:, :seq_len - 1, :]
        decode_token = x[:, seq_len - 1:, :]
        seq_ids = torch.tensor([seq_id], dtype=torch.long)

        _ = mla(prefill, paged_kv_cache=paged, layer_idx=0, batch_indices=seq_ids)
        cached_out = mla(
            decode_token, paged_kv_cache=paged, layer_idx=0, batch_indices=seq_ids,
        )

    torch.testing.assert_close(
        cached_out[:, 0, :], ref_out[:, -1, :], atol=1e-5, rtol=1e-5,
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
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
        num_blocks=4, block_size=4,
    )
    x = torch.randn(1, 1, hidden_size)

    with pytest.raises(ValueError, match="layer_idx is required"):
        mla(
            x, paged_kv_cache=paged, layer_idx=None,
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
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim, block_size=4,
    )

    # Prefill 5 tokens → 2 blocks.
    prefill = torch.randn(1, 5, hidden_size)
    seq_ids = torch.tensor([seq_id], dtype=torch.long)
    mla(prefill, paged_kv_cache=paged, layer_idx=0, batch_indices=seq_ids)
    blocks_after_prefill = paged.get_block_table(seq_id)
    assert len(blocks_after_prefill) == 2

    # Decode one token → same block table, count = 6.
    decode = torch.randn(1, 1, hidden_size)
    mla(decode, paged_kv_cache=paged, layer_idx=0, batch_indices=seq_ids)
    assert paged.get_block_table(seq_id) == blocks_after_prefill
    assert paged.block_manager.get_num_tokens(seq_id) == 6
