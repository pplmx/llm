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
