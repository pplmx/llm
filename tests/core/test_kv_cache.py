"""Tests for KVCache pre-allocated cache manager."""

import pytest
import torch

from llm.core.kv_cache import KVCache, reset_all_caches
from tests.support.devices import ALL_DEVICES, DEFAULT_DEVICE


class TestKVCache:
    """Tests for KVCache class."""

    @pytest.mark.parametrize("device", ALL_DEVICES)
    def test_initialization(self, device):
        """Test cache initialization with correct shapes."""
        cache = KVCache(
            max_batch_size=2,
            max_seq_len=128,
            num_kv_heads=4,
            head_dim=32,
            device=device,
            dtype=torch.float32,
        )

        assert cache.k_cache.shape == (2, 4, 128, 32)
        assert cache.v_cache.shape == (2, 4, 128, 32)
        assert cache.seq_len == 0

    @pytest.mark.parametrize("device", ALL_DEVICES)
    def test_update_single_token(self, device):
        """Test updating cache with single token."""
        cache = KVCache(2, 128, 4, 32, device=device, dtype=torch.float32)

        k_new = torch.randn(2, 4, 1, 32, device=device)
        v_new = torch.randn(2, 4, 1, 32, device=device)

        k_out, v_out = cache.update(k_new, v_new)

        assert k_out.shape == (2, 4, 1, 32)
        assert v_out.shape == (2, 4, 1, 32)
        assert cache.seq_len == 1
        assert torch.allclose(k_out, k_new)
        assert torch.allclose(v_out, v_new)

    @pytest.mark.parametrize("device", ALL_DEVICES)
    def test_update_multiple_steps(self, device):
        """Test sequential updates accumulate correctly."""
        cache = KVCache(1, 128, 2, 16, device=device, dtype=torch.float32)

        # First update: prompt with 10 tokens
        k1 = torch.randn(1, 2, 10, 16, device=device)
        v1 = torch.randn(1, 2, 10, 16, device=device)
        k_out, _v_out = cache.update(k1, v1)

        assert cache.seq_len == 10
        assert k_out.shape == (1, 2, 10, 16)

        # Second update: 1 new token
        k2 = torch.randn(1, 2, 1, 16, device=device)
        v2 = torch.randn(1, 2, 1, 16, device=device)
        k_out, _v_out = cache.update(k2, v2)

        assert cache.seq_len == 11
        assert k_out.shape == (1, 2, 11, 16)

        # Verify first 10 tokens are preserved
        assert torch.allclose(k_out[:, :, :10], k1)
        assert torch.allclose(k_out[:, :, 10:11], k2)

    @pytest.mark.parametrize("device", ALL_DEVICES)
    def test_reset(self, device):
        """Test cache reset clears sequence length."""
        cache = KVCache(1, 64, 2, 16, device=device, dtype=torch.float32)

        cache.update(torch.randn(1, 2, 5, 16, device=device), torch.randn(1, 2, 5, 16, device=device))
        assert cache.seq_len == 5

        cache.reset()
        assert cache.seq_len == 0

    @pytest.mark.parametrize("device", ALL_DEVICES)
    def test_overflow_raises_error(self, device):
        """Test that exceeding max_seq_len raises ValueError."""
        cache = KVCache(1, 10, 2, 16, device=device, dtype=torch.float32)

        # Fill to capacity
        cache.update(torch.randn(1, 2, 10, 16, device=device), torch.randn(1, 2, 10, 16, device=device))

        # Try to add one more
        with pytest.raises(ValueError, match="Cache overflow"):
            cache.update(torch.randn(1, 2, 1, 16, device=device), torch.randn(1, 2, 1, 16, device=device))

    @pytest.mark.parametrize("device", ALL_DEVICES)
    def test_from_model_config(self, device):
        """Test factory method creates correct number of caches."""
        caches = KVCache.from_model_config(
            max_batch_size=4,
            max_seq_len=256,
            num_layers=12,
            num_kv_heads=8,
            head_dim=64,
            device=device,
            dtype=torch.float32,
        )

        assert len(caches) == 12
        for cache in caches:
            assert cache.k_cache.shape == (4, 8, 256, 64)

    @pytest.mark.parametrize("device", ALL_DEVICES)
    def test_reset_all_caches(self, device):
        """Test reset_all_caches utility function."""
        caches = KVCache.from_model_config(2, 64, 3, 4, 16, device, torch.float32)

        for cache in caches:
            cache.update(torch.randn(2, 4, 5, 16, device=device), torch.randn(2, 4, 5, 16, device=device))
            assert cache.seq_len == 5

        reset_all_caches(caches)

        for cache in caches:
            assert cache.seq_len == 0

    def test_device_dtype_properties(self):
        """Test device and dtype properties."""
        cache = KVCache(1, 32, 2, 16, device=str(DEFAULT_DEVICE), dtype=torch.float16)

        assert cache.device == DEFAULT_DEVICE
        assert cache.dtype == torch.float16

    @pytest.mark.parametrize("device", ALL_DEVICES)
    def test_no_memory_allocation_on_update(self, device):
        """Test that update does not allocate new memory."""
        cache = KVCache(1, 64, 2, 16, device=device, dtype=torch.float32)

        # Get initial data_ptr
        k_ptr = cache.k_cache.data_ptr()
        v_ptr = cache.v_cache.data_ptr()

        # Perform updates
        for _ in range(10):
            cache.update(torch.randn(1, 2, 1, 16, device=device), torch.randn(1, 2, 1, 16, device=device))

        # Verify buffers are the same (no reallocation)
        assert cache.k_cache.data_ptr() == k_ptr
        assert cache.v_cache.data_ptr() == v_ptr

    @pytest.mark.parametrize("device", ALL_DEVICES)
    def test_update_at_indices_decode_writes_each_slot_own_position(self, device):
        """Decode writes must land only on each slot's own position.

        Regression: ``start_pos`` arrives as ``[B, 1]``; using it
        unflattened broadcasts with ``batch_indices`` into a [B, B]
        index grid and every slot's K/V was written at every batch
        position, corrupting unrelated cache entries.
        """
        cache = KVCache(4, 128, 1, 8, device=device, dtype=torch.float32)
        batch_indices = torch.tensor([0, 1, 2, 3])
        start_pos = torch.tensor([[4], [11], [2], [6]])  # [B, 1] as position_ids
        k_new = torch.randn(4, 1, 1, 8, device=device)
        v_new = torch.randn(4, 1, 1, 8, device=device)

        cache.update_at_indices(batch_indices, k_new, v_new, start_pos)

        for slot, pos in enumerate([4, 11, 2, 6]):
            assert torch.allclose(cache.k_cache[slot, 0, pos], k_new[slot, 0, 0])
            assert torch.allclose(cache.v_cache[slot, 0, pos], v_new[slot, 0, 0])
        # Off-diagonal slots must remain untouched (zero).
        for slot, pos in enumerate([4, 11, 2, 6]):
            for other in range(4):
                other_pos = [4, 11, 2, 6][other]
                if other_pos != pos:
                    assert torch.all(cache.k_cache[slot, 0, other_pos] == 0), (slot, other_pos)

    @pytest.mark.parametrize("device", ALL_DEVICES)
    def test_update_at_indices_prefill_keeps_real_position_zero(self, device):
        """Padded slots (position_id 0) must not overwrite a short row's
        real position-0 K/V during a mixed-length prefill."""
        cache = KVCache(2, 128, 1, 8, device=device, dtype=torch.float32)
        batch_indices = torch.tensor([0, 1])
        # Row 0: real length 2 (positions 0,1), padded positions reuse 0.
        # Row 1: full length 4.
        start_pos = torch.tensor([[0, 1, 0, 0], [0, 1, 2, 3]])
        k_new = torch.randn(2, 1, 4, 8, device=device)

        cache.update_at_indices(batch_indices, k_new, k_new.clone(), start_pos)

        # Row 0's position 0 must keep the REAL K (k_new[0,:,0,:]), not the
        # padded slot's K (k_new[0,:,2,:] == k_new[0,:,3,:]).
        assert torch.allclose(cache.k_cache[0, 0, 0], k_new[0, 0, 0])
        assert torch.allclose(cache.k_cache[0, 0, 1], k_new[0, 0, 1])
        assert torch.allclose(cache.k_cache[1, 0, :4], k_new[1, 0, :4])

    @pytest.mark.parametrize("device", ALL_DEVICES)
    def test_update_at_indices_decode_row_does_not_clobber_cached_position_zero(self, device):
        """A decode row's padded position-0 columns must not overwrite the
        genuine position-0 K/V cached during that slot's earlier prefill.

        Regression: in a mixed batch the decode row's real write is at
        ``start_pos[b, 0]`` (its ``total_len - 1``), and the padded columns
        all reuse position 0. The previous dedup-keep-first logic treated the
        first ``(slot, 0)`` occurrence as real; for a decode row that is the
        pad column, silently corrupting the cached first token for every
        later query.
        """
        cache = KVCache(2, 16, 1, 8, device=device, dtype=torch.float32)
        batch = torch.tensor([0, 1], device=device)
        # Prefill both slots with 5 real tokens at positions 0..4.
        k5 = torch.randn(2, 1, 5, 8, device=device)
        v5 = torch.randn(2, 1, 5, 8, device=device)
        start5 = torch.arange(5, device=device).view(1, 5).expand(2, 5)
        cache.update_at_indices(batch, k5, v5, start5)
        real_p0 = cache.k_cache[0, 0, 0].clone()

        # Mixed step: both rows decode one token at position 5, padded
        # columns reuse position 0.
        mix_k = torch.randn(2, 1, 3, 8, device=device)
        mix_v = torch.randn(2, 1, 3, 8, device=device)
        mix_pos = torch.tensor([[5, 0, 0], [5, 0, 0]], device=device)
        cache.update_at_indices(batch, mix_k, mix_v, mix_pos)

        # Genuine position-0 K/V survives, and the real decode writes land
        # at position 5 only.
        assert torch.allclose(cache.k_cache[0, 0, 0], real_p0)
        assert torch.allclose(cache.k_cache[1, 0, 0], k5[1, 0, 0])
        assert torch.allclose(cache.k_cache[0, 0, 5], mix_k[0, 0, 0])
        assert torch.allclose(cache.k_cache[1, 0, 5], mix_k[1, 0, 0])
        # No pad column leaked anywhere else (nothing written beyond position 5).
        assert torch.all(cache.k_cache[0, 0, 6:] == 0)
        assert torch.all(cache.k_cache[1, 0, 6:] == 0)

    @pytest.mark.parametrize("device", ALL_DEVICES)
    def test_update_at_indices_decode_near_max_len_does_not_overflow_with_longer_prefill(self, device):
        """Overflow must be judged per-row (real write end), not by the
        batch-max ``seq_len_new``.

        Regression: a decode row appending one token at
        ``start_pos[b,0] == max_seq_len - 1`` is valid, but the old check
        ``start_pos[:, 0] + seq_len_new > max_seq_len`` added the batch-max
        prefill length and raised a spurious ``Cache overflow`` (HTTP 500 in
        the serving engine).
        """
        max_seq_len = 8
        batch = torch.tensor([0, 1], device=device)
        cache = KVCache(2, max_seq_len, 1, 8, device=device, dtype=torch.float32)
        # Prefill both slots 4 tokens (0..3).
        k4 = torch.randn(2, 1, 4, 8, device=device)
        cache.update_at_indices(batch, k4, k4.clone(), torch.arange(4, device=device).view(1, 4).expand(2, 4))
        # Next step: row 0 prefill 4 new tokens at 4..7; row 1 decode one
        # token at position 6 (valid: 6 < 8).
        mix_k = torch.randn(2, 1, 4, 8, device=device)
        mix_pos = torch.tensor([[4, 5, 6, 7], [6, 0, 0, 0]], device=device)
        cache.update_at_indices(batch, mix_k, mix_k.clone(), mix_pos)

        assert torch.allclose(cache.k_cache[0, 0, 4:8], mix_k[0, 0, :4])
        assert torch.allclose(cache.k_cache[1, 0, 6], mix_k[1, 0, 0])

        # The same decode row would genuinely overflow by appending at >= 8.
        over_pos = torch.tensor([[4, 5, 6, 7], [8, 0, 0, 0]], device=device)
        with pytest.raises(ValueError, match="Cache overflow"):
            cache.update_at_indices(batch, mix_k, mix_k.clone(), over_pos)
