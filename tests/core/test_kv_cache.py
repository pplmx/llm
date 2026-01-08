"""Tests for KVCache pre-allocated cache manager."""

import pytest
import torch

from llm.core.kv_cache import KVCache, reset_all_caches


class TestKVCache:
    """Tests for KVCache class."""

    def test_initialization(self):
        """Test cache initialization with correct shapes."""
        cache = KVCache(
            max_batch_size=2,
            max_seq_len=128,
            num_kv_heads=4,
            head_dim=32,
            device="cpu",
            dtype=torch.float32,
        )

        assert cache.k_cache.shape == (2, 4, 128, 32)
        assert cache.v_cache.shape == (2, 4, 128, 32)
        assert cache.seq_len == 0

    def test_update_single_token(self):
        """Test updating cache with single token."""
        cache = KVCache(2, 128, 4, 32, device="cpu", dtype=torch.float32)

        k_new = torch.randn(2, 4, 1, 32)
        v_new = torch.randn(2, 4, 1, 32)

        k_out, v_out = cache.update(k_new, v_new)

        assert k_out.shape == (2, 4, 1, 32)
        assert v_out.shape == (2, 4, 1, 32)
        assert cache.seq_len == 1
        assert torch.allclose(k_out, k_new)
        assert torch.allclose(v_out, v_new)

    def test_update_multiple_steps(self):
        """Test sequential updates accumulate correctly."""
        cache = KVCache(1, 128, 2, 16, device="cpu", dtype=torch.float32)

        # First update: prompt with 10 tokens
        k1 = torch.randn(1, 2, 10, 16)
        v1 = torch.randn(1, 2, 10, 16)
        k_out, v_out = cache.update(k1, v1)

        assert cache.seq_len == 10
        assert k_out.shape == (1, 2, 10, 16)

        # Second update: 1 new token
        k2 = torch.randn(1, 2, 1, 16)
        v2 = torch.randn(1, 2, 1, 16)
        k_out, v_out = cache.update(k2, v2)

        assert cache.seq_len == 11
        assert k_out.shape == (1, 2, 11, 16)

        # Verify first 10 tokens are preserved
        assert torch.allclose(k_out[:, :, :10], k1)
        assert torch.allclose(k_out[:, :, 10:11], k2)

    def test_reset(self):
        """Test cache reset clears sequence length."""
        cache = KVCache(1, 64, 2, 16, device="cpu", dtype=torch.float32)

        cache.update(torch.randn(1, 2, 5, 16), torch.randn(1, 2, 5, 16))
        assert cache.seq_len == 5

        cache.reset()
        assert cache.seq_len == 0

    def test_overflow_raises_error(self):
        """Test that exceeding max_seq_len raises ValueError."""
        cache = KVCache(1, 10, 2, 16, device="cpu", dtype=torch.float32)

        # Fill to capacity
        cache.update(torch.randn(1, 2, 10, 16), torch.randn(1, 2, 10, 16))

        # Try to add one more
        with pytest.raises(ValueError, match="Cache overflow"):
            cache.update(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))

    def test_from_model_config(self):
        """Test factory method creates correct number of caches."""
        caches = KVCache.from_model_config(
            max_batch_size=4,
            max_seq_len=256,
            num_layers=12,
            num_kv_heads=8,
            head_dim=64,
            device="cpu",
            dtype=torch.float32,
        )

        assert len(caches) == 12
        for cache in caches:
            assert cache.k_cache.shape == (4, 8, 256, 64)

    def test_reset_all_caches(self):
        """Test reset_all_caches utility function."""
        caches = KVCache.from_model_config(2, 64, 3, 4, 16, "cpu", torch.float32)

        for cache in caches:
            cache.update(torch.randn(2, 4, 5, 16), torch.randn(2, 4, 5, 16))
            assert cache.seq_len == 5

        reset_all_caches(caches)

        for cache in caches:
            assert cache.seq_len == 0

    def test_device_dtype_properties(self):
        """Test device and dtype properties."""
        cache = KVCache(1, 32, 2, 16, device="cpu", dtype=torch.float16)

        assert cache.device == torch.device("cpu")
        assert cache.dtype == torch.float16

    def test_no_memory_allocation_on_update(self):
        """Test that update does not allocate new memory."""
        cache = KVCache(1, 64, 2, 16, device="cpu", dtype=torch.float32)

        # Get initial data_ptr
        k_ptr = cache.k_cache.data_ptr()
        v_ptr = cache.v_cache.data_ptr()

        # Perform updates
        for _ in range(10):
            cache.update(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))

        # Verify buffers are the same (no reallocation)
        assert cache.k_cache.data_ptr() == k_ptr
        assert cache.v_cache.data_ptr() == v_ptr
