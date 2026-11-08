"""Tests for Paged Attention components."""

import pytest
import torch

from llm.core.paged_attention.attention import paged_attention_forward
from llm.core.paged_attention.block_allocator import BlockAllocator
from llm.core.paged_attention.block_manager import BlockManager


class TestBlockAllocator:
    """Tests for BlockAllocator."""

    def test_init(self):
        """Test allocator initialization."""
        allocator = BlockAllocator(num_blocks=100, block_size=16)

        assert allocator.num_blocks == 100
        assert allocator.block_size == 16
        assert allocator.num_free_blocks == 100
        assert allocator.num_allocated_blocks == 0

    def test_allocate_single(self):
        """Test single block allocation."""
        allocator = BlockAllocator(num_blocks=10)

        block_id = allocator.allocate()

        assert block_id == 0
        assert allocator.num_free_blocks == 9
        assert allocator.get_ref_count(block_id) == 1

    def test_allocate_multiple(self):
        """Test multiple block allocation."""
        allocator = BlockAllocator(num_blocks=10)

        blocks = allocator.allocate_n(5)

        assert len(blocks) == 5
        assert allocator.num_free_blocks == 5

    def test_allocate_exhaustion(self):
        """Test allocation when no blocks available."""
        allocator = BlockAllocator(num_blocks=2)

        allocator.allocate()
        allocator.allocate()

        with pytest.raises(RuntimeError, match="No free blocks"):
            allocator.allocate()

    def test_free_block(self):
        """Test block freeing."""
        allocator = BlockAllocator(num_blocks=10)

        block_id = allocator.allocate()
        assert allocator.num_free_blocks == 9

        allocator.free(block_id)
        assert allocator.num_free_blocks == 10

    def test_fork_increments_refcount(self):
        """Test fork increments reference count."""
        allocator = BlockAllocator(num_blocks=10)

        block_id = allocator.allocate()
        assert allocator.get_ref_count(block_id) == 1

        forked_id = allocator.fork(block_id)
        assert forked_id == block_id  # Same block
        assert allocator.get_ref_count(block_id) == 2
        assert allocator.is_shared(block_id)

    def test_copy_on_write_shared(self):
        """Test copy-on-write for shared blocks."""
        allocator = BlockAllocator(num_blocks=10)

        block_id = allocator.allocate()
        allocator.fork(block_id)  # Now ref_count = 2

        new_block_id = allocator.copy_on_write(block_id)

        assert new_block_id != block_id
        assert allocator.get_ref_count(block_id) == 1
        assert allocator.get_ref_count(new_block_id) == 1

    def test_copy_on_write_unshared(self):
        """Test copy-on-write returns same block if not shared."""
        allocator = BlockAllocator(num_blocks=10)

        block_id = allocator.allocate()
        assert allocator.get_ref_count(block_id) == 1

        same_id = allocator.copy_on_write(block_id)
        assert same_id == block_id

    def test_reset(self):
        """Test allocator reset."""
        allocator = BlockAllocator(num_blocks=10)

        allocator.allocate_n(5)
        assert allocator.num_free_blocks == 5

        allocator.reset()
        assert allocator.num_free_blocks == 10


class TestBlockManager:
    """Tests for BlockManager."""

    def test_init(self):
        """Test block manager initialization."""
        manager = BlockManager(num_blocks=100, block_size=16, num_layers=4)

        assert manager.block_size == 16
        assert manager.num_layers == 4
        assert len(manager.allocators) == 4

    def test_allocate_sequence(self):
        """Test sequence allocation."""
        manager = BlockManager(num_blocks=100, block_size=16, num_layers=2)

        blocks = manager.allocate_sequence(seq_id=1, num_tokens=32)

        assert len(blocks) == 2  # 32 tokens / 16 block_size = 2 blocks
        assert manager.get_num_tokens(1) == 32
        assert 1 in manager.get_all_sequence_ids()

    def test_extend_sequence(self):
        """Test sequence extension."""
        manager = BlockManager(num_blocks=100, block_size=16, num_layers=1)

        manager.allocate_sequence(seq_id=1, num_tokens=10)
        assert len(manager.get_block_table(1)) == 1

        # Extend within same block
        blocks = manager.extend_sequence(seq_id=1, num_new_tokens=5)
        assert len(blocks) == 1  # Still fits in one block
        assert manager.get_num_tokens(1) == 15

        # Extend to need new block
        blocks = manager.extend_sequence(seq_id=1, num_new_tokens=10)
        assert len(blocks) == 2  # Now needs 2 blocks
        assert manager.get_num_tokens(1) == 25

    def test_free_sequence(self):
        """Test sequence freeing."""
        manager = BlockManager(num_blocks=10, block_size=16, num_layers=1)

        initial_free = manager.num_free_blocks
        manager.allocate_sequence(seq_id=1, num_tokens=32)
        assert manager.num_free_blocks < initial_free

        manager.free_sequence(seq_id=1)
        assert manager.num_free_blocks == initial_free
        assert 1 not in manager.get_all_sequence_ids()

    def test_fork_sequence(self):
        """Test sequence forking (copy-on-write)."""
        manager = BlockManager(num_blocks=100, block_size=16, num_layers=1)

        manager.allocate_sequence(seq_id=1, num_tokens=32)
        original_blocks = manager.get_block_table(1)

        forked_blocks = manager.fork_sequence(src_seq_id=1, dst_seq_id=2)

        assert forked_blocks == original_blocks
        assert manager.get_num_tokens(2) == 32
        assert 2 in manager.get_all_sequence_ids()

    def test_can_allocate_sequence(self):
        """Test allocation check."""
        manager = BlockManager(num_blocks=5, block_size=16, num_layers=1)

        assert manager.can_allocate_sequence(num_tokens=16)  # 1 block
        assert manager.can_allocate_sequence(num_tokens=80)  # 5 blocks
        assert not manager.can_allocate_sequence(num_tokens=100)  # 7 blocks

    def test_multiple_sequences(self):
        """Test managing multiple sequences."""
        manager = BlockManager(num_blocks=100, block_size=16, num_layers=1)

        manager.allocate_sequence(seq_id=1, num_tokens=32)
        manager.allocate_sequence(seq_id=2, num_tokens=48)
        manager.allocate_sequence(seq_id=3, num_tokens=16)

        assert len(manager.get_all_sequence_ids()) == 3
        assert manager.get_num_tokens(1) == 32
        assert manager.get_num_tokens(2) == 48
        assert manager.get_num_tokens(3) == 16

    def test_extend_and_free_multi_layer(self):
        """Test that blocks are properly freed after extend + free with multiple layers."""
        manager = BlockManager(num_blocks=100, block_size=16, num_layers=3)

        initial_free = manager.num_free_blocks
        manager.allocate_sequence(seq_id=1, num_tokens=10)
        assert manager.num_free_blocks < initial_free

        # Extend to need new blocks
        manager.extend_sequence(seq_id=1, num_new_tokens=30)

        # Free and verify all blocks returned
        manager.free_sequence(seq_id=1)
        assert manager.num_free_blocks == initial_free

        # Verify allocators are consistent
        for alloc in manager.allocators:
            assert alloc.num_allocated_blocks == 0

    def test_fork_sequence_multi_layer(self):
        """Test fork with multiple layers keeps allocators consistent."""
        manager = BlockManager(num_blocks=100, block_size=16, num_layers=3)

        manager.allocate_sequence(seq_id=1, num_tokens=16)
        forked_blocks = manager.fork_sequence(src_seq_id=1, dst_seq_id=2)

        assert forked_blocks == manager.get_block_table(1)
        assert manager.get_num_tokens(2) == 16

        # Both sequences should share blocks with ref_count == 2
        for alloc in manager.allocators:
            assert alloc.get_ref_count(forked_blocks[0]) == 2

        # Freeing one should not affect the other (ref_count goes to 1)
        manager.free_sequence(seq_id=1)
        for alloc in manager.allocators:
            assert alloc.get_ref_count(forked_blocks[0]) == 1
        assert 1 not in manager.get_all_sequence_ids()
        assert 2 in manager.get_all_sequence_ids()

        # Freeing the other should return all blocks
        manager.free_sequence(seq_id=2)
        assert manager.num_free_blocks == 100

    def test_extend_insufficient_blocks(self):
        """Test that extend raises when not enough blocks available."""
        manager = BlockManager(num_blocks=2, block_size=4, num_layers=1)

        manager.allocate_sequence(seq_id=1, num_tokens=4)  # 1 block
        assert manager.num_free_blocks == 1

        # Trying to extend by 5 tokens needs another block but only 1 is free
        with pytest.raises(RuntimeError):
            manager.extend_sequence(seq_id=1, num_new_tokens=5)

    def test_free_nonexistent_is_noop(self):
        """Test that freeing a non-existent sequence is a no-op."""
        manager = BlockManager(num_blocks=10, block_size=16, num_layers=1)

        # Should not raise
        manager.free_sequence(seq_id=999)
        assert manager.num_free_blocks == 10

    def test_get_block_table_unknown_raises(self):
        """Test that get_block_table raises for unknown sequence."""
        manager = BlockManager(num_blocks=10, block_size=16, num_layers=1)

        with pytest.raises(ValueError, match="does not exist"):
            manager.get_block_table(seq_id=999)

    def test_get_num_tokens_unknown_raises(self):
        """Test that get_num_tokens raises for unknown sequence."""
        manager = BlockManager(num_blocks=10, block_size=16, num_layers=1)

        with pytest.raises(ValueError, match="does not exist"):
            manager.get_num_tokens(seq_id=999)

    def test_allocate_existing_sequence_raises(self):
        """Test that allocating an existing sequence raises."""
        manager = BlockManager(num_blocks=10, block_size=16, num_layers=1)

        manager.allocate_sequence(seq_id=1, num_tokens=16)
        with pytest.raises(ValueError, match="already exists"):
            manager.allocate_sequence(seq_id=1, num_tokens=16)

    def test_fork_nonexistent_source_raises(self):
        """Test that forking from non-existent source raises."""
        manager = BlockManager(num_blocks=10, block_size=16, num_layers=1)

        with pytest.raises(ValueError, match="does not exist"):
            manager.fork_sequence(src_seq_id=999, dst_seq_id=1)

    def test_fork_to_existing_dest_raises(self):
        """Test that forking to existing destination raises."""
        manager = BlockManager(num_blocks=10, block_size=16, num_layers=1)

        manager.allocate_sequence(seq_id=1, num_tokens=16)
        manager.allocate_sequence(seq_id=2, num_tokens=16)

        with pytest.raises(ValueError, match="already exists"):
            manager.fork_sequence(src_seq_id=1, dst_seq_id=2)

    def test_extend_nonexistent_raises(self):
        """Test that extending a non-existent sequence raises."""
        manager = BlockManager(num_blocks=10, block_size=16, num_layers=1)

        with pytest.raises(ValueError, match="does not exist"):
            manager.extend_sequence(seq_id=999, num_new_tokens=5)

    def test_repr(self):
        """Test __repr__ output."""
        manager = BlockManager(num_blocks=100, block_size=16, num_layers=2)
        repr_str = repr(manager)
        assert "BlockManager" in repr_str
        assert "block_size=16" in repr_str
        assert "num_layers=2" in repr_str

    def test_can_allocate_zero_tokens(self):
        """Test that zero tokens can always be allocated."""
        manager = BlockManager(num_blocks=10, block_size=16, num_layers=1)
        assert manager.can_allocate_sequence(num_tokens=0)

    def test_can_allocate_negative_tokens(self):
        """Test that negative tokens can always be allocated."""
        manager = BlockManager(num_blocks=10, block_size=16, num_layers=1)
        assert manager.can_allocate_sequence(num_tokens=-5)

    def test_reset_with_multiple_sequences(self):
        """Test reset with multiple sequences and layers."""
        manager = BlockManager(num_blocks=100, block_size=16, num_layers=3)

        manager.allocate_sequence(seq_id=1, num_tokens=16)
        manager.allocate_sequence(seq_id=2, num_tokens=32)
        manager.fork_sequence(src_seq_id=1, dst_seq_id=3)

        assert len(manager.get_all_sequence_ids()) == 3
        manager.reset()
        assert len(manager.get_all_sequence_ids()) == 0
        assert manager.num_free_blocks == 100

        for alloc in manager.allocators:
            assert alloc.num_allocated_blocks == 0

    def test_reset(self):
        """Test manager reset."""
        manager = BlockManager(num_blocks=100, block_size=16, num_layers=1)

        manager.allocate_sequence(seq_id=1, num_tokens=32)
        manager.allocate_sequence(seq_id=2, num_tokens=48)

        manager.reset()

        assert len(manager.get_all_sequence_ids()) == 0
        assert manager.num_free_blocks == 100


class TestPagedAttentionForward:
    """Tests for paged_attention_forward."""

    def test_paged_attention_output_shape(self):
        """Test paged attention output has correct shape."""
        batch_size = 2
        num_heads = 4
        head_dim = 16
        block_size = 16
        num_kv_heads = 2
        num_blocks = 8

        q = torch.randn(batch_size, num_heads, 1, head_dim)

        k_cache = torch.randn(1, num_blocks, num_kv_heads, block_size, head_dim)
        v_cache = torch.randn(1, num_blocks, num_kv_heads, block_size, head_dim)

        block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        seq_lens = torch.tensor([20, 25])

        output = paged_attention_forward(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=num_kv_heads,
        )

        assert output.shape == (batch_size, num_heads, 1, head_dim)

    def test_single_sequence(self):
        """Test with single sequence in batch."""
        batch_size = 1
        num_heads = 4
        head_dim = 16
        num_kv_heads = 2

        q = torch.randn(batch_size, num_heads, 1, head_dim)
        k_cache = torch.randn(1, 4, num_kv_heads, 16, head_dim)
        v_cache = torch.randn(1, 4, num_kv_heads, 16, head_dim)

        block_tables = torch.tensor([[0, 1]], dtype=torch.long)
        seq_lens = torch.tensor([20])

        output = paged_attention_forward(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=num_kv_heads,
        )

        assert output.shape == (1, num_heads, 1, head_dim)

    def test_gqa_expansion(self):
        """Test GQA (Grouped Query Attention) with num_kv_heads != num_heads."""
        batch_size = 1
        num_heads = 8
        num_kv_heads = 2  # 4x expansion
        head_dim = 16

        q = torch.randn(batch_size, num_heads, 1, head_dim)
        k_cache = torch.randn(1, 4, num_kv_heads, 16, head_dim)
        v_cache = torch.randn(1, 4, num_kv_heads, 16, head_dim)

        block_tables = torch.tensor([[0, 1]], dtype=torch.long)
        seq_lens = torch.tensor([20])

        output = paged_attention_forward(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=num_kv_heads,
        )

        assert output.shape == (batch_size, num_heads, 1, head_dim)

    def test_equal_heads(self):
        """Test with num_kv_heads == num_heads (no GQA)."""
        batch_size = 1
        num_heads = 4
        num_kv_heads = 4
        head_dim = 16

        q = torch.randn(batch_size, num_heads, 1, head_dim)
        k_cache = torch.randn(1, 4, num_kv_heads, 16, head_dim)
        v_cache = torch.randn(1, 4, num_kv_heads, 16, head_dim)

        block_tables = torch.tensor([[0, 1]], dtype=torch.long)
        seq_lens = torch.tensor([20])

        output = paged_attention_forward(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=num_kv_heads,
        )

        assert output.shape == (batch_size, num_heads, 1, head_dim)

    def test_different_seq_lens(self):
        """Test with sequences of different lengths."""
        batch_size = 3
        num_heads = 4
        head_dim = 16
        num_kv_heads = 2

        q = torch.randn(batch_size, num_heads, 1, head_dim)
        k_cache = torch.randn(1, 6, num_kv_heads, 16, head_dim)
        v_cache = torch.randn(1, 6, num_kv_heads, 16, head_dim)

        # Different lengths: 10, 32, 50 tokens
        block_tables = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
        seq_lens = torch.tensor([10, 32, 50])

        output = paged_attention_forward(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=num_kv_heads,
        )

        assert output.shape == (batch_size, num_heads, 1, head_dim)

    def test_single_token_sequence(self):
        """Test with single token sequence."""
        batch_size = 1
        num_heads = 4
        head_dim = 16
        num_kv_heads = 2

        q = torch.randn(batch_size, num_heads, 1, head_dim)
        k_cache = torch.randn(1, 1, num_kv_heads, 16, head_dim)
        v_cache = torch.randn(1, 1, num_kv_heads, 16, head_dim)

        block_tables = torch.tensor([[0]], dtype=torch.long)
        seq_lens = torch.tensor([1])

        output = paged_attention_forward(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=num_kv_heads,
        )

        assert output.shape == (batch_size, num_heads, 1, head_dim)

    def test_large_block_size(self):
        """Test with large block size."""
        batch_size = 1
        num_heads = 4
        head_dim = 16
        num_kv_heads = 2
        block_size = 64

        q = torch.randn(batch_size, num_heads, 1, head_dim)
        k_cache = torch.randn(1, 2, num_kv_heads, block_size, head_dim)
        v_cache = torch.randn(1, 2, num_kv_heads, block_size, head_dim)

        block_tables = torch.tensor([[0, 1]], dtype=torch.long)
        seq_lens = torch.tensor([100])

        output = paged_attention_forward(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
        )

        assert output.shape == (batch_size, num_heads, 1, head_dim)

    def test_multi_layer_cache(self):
        """Test with multi-layer KV cache."""
        batch_size = 1
        num_heads = 4
        head_dim = 16
        num_kv_heads = 2
        num_layers = 4

        q = torch.randn(batch_size, num_heads, 1, head_dim)
        k_cache = torch.randn(num_layers, 4, num_kv_heads, 16, head_dim)
        v_cache = torch.randn(num_layers, 4, num_kv_heads, 16, head_dim)

        block_tables = torch.tensor([[0, 1]], dtype=torch.long)
        seq_lens = torch.tensor([20])

        output = paged_attention_forward(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=num_kv_heads,
        )

        assert output.shape == (batch_size, num_heads, 1, head_dim)

    def test_output_values_are_finite(self):
        """Test that output values are finite (no NaN/Inf)."""
        batch_size = 2
        num_heads = 4
        head_dim = 32
        num_kv_heads = 2

        q = torch.randn(batch_size, num_heads, 1, head_dim)
        k_cache = torch.randn(1, 4, num_kv_heads, 16, head_dim) * 0.1  # Smaller values
        v_cache = torch.randn(1, 4, num_kv_heads, 16, head_dim) * 0.1

        block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        seq_lens = torch.tensor([20, 25])

        output = paged_attention_forward(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=num_kv_heads,
        )

        assert torch.isfinite(output).all()

    def test_prefill_multi_query_tokens(self):
        """Prefill: ``q`` carries ``S_q > 1`` query tokens per row.

        For prefill the model attends each query token to the full
        cached context of its sequence. The kernel already gathered
        the whole k/v slice per row — the multi-token generalisation
        just lets the matmul produce ``S_q`` outputs.
        """
        batch_size = 2
        num_heads = 4
        head_dim = 16
        block_size = 16
        num_kv_heads = 2
        num_blocks = 8
        query_len = 5  # prefill-style: 5 query tokens per row

        q = torch.randn(batch_size, num_heads, query_len, head_dim)
        k_cache = torch.randn(1, num_blocks, num_kv_heads, block_size, head_dim)
        v_cache = torch.randn(1, num_blocks, num_kv_heads, block_size, head_dim)

        block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        seq_lens = torch.tensor([20, 25])

        output = paged_attention_forward(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=num_kv_heads,
        )

        assert output.shape == (batch_size, num_heads, query_len, head_dim)
        assert torch.isfinite(output).all()

    def test_prefill_matches_decode_per_token(self):
        """Each prefill output token equals a single decode forward over the same context.

        Loops one query token at a time and concatenates; this must
        match a single ``query_len``-shot prefill call (within fp32
        tolerance). Pins the multi-token generalisation to the
        single-token reference behaviour.
        """
        torch.manual_seed(0)
        batch_size = 1
        num_heads = 2
        head_dim = 8
        block_size = 16
        num_kv_heads = 2
        num_blocks = 4
        query_len = 3

        q = torch.randn(batch_size, num_heads, query_len, head_dim)
        k_cache = torch.randn(1, num_blocks, num_kv_heads, block_size, head_dim)
        v_cache = torch.randn(1, num_blocks, num_kv_heads, block_size, head_dim)
        block_tables = torch.tensor([[0, 1]], dtype=torch.long)
        # Sequence is exactly ``query_len`` tokens so the prefill covers
        # the full context (no past beyond the new tokens themselves).
        seq_lens = torch.tensor([query_len])

        prefill_out = paged_attention_forward(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=num_kv_heads,
        )

        # Decode reference: one query token at a time. A multi-token
        # prefill is causal — query row ``t`` attends to keys ``0..t`` —
        # so each decode step sees only the context written so far
        # (``seq_lens = t + 1``), not the full future context.
        decode_outs = [
            paged_attention_forward(
                q=q[:, :, t : t + 1, :],
                k_cache=k_cache,
                v_cache=v_cache,
                block_tables=block_tables,
                seq_lens=torch.tensor([t + 1]),
                num_kv_heads=num_kv_heads,
            )
            for t in range(query_len)
        ]
        decode_out = torch.cat(decode_outs, dim=2)

        assert prefill_out.shape == decode_out.shape
        assert torch.allclose(prefill_out, decode_out, atol=1e-5)

    def test_zero_seq_len(self):
        """Edge case: seq_len == 0 should not crash (returns zeros)."""
        batch_size = 1
        num_heads = 4
        head_dim = 16
        num_kv_heads = 2

        q = torch.randn(batch_size, num_heads, 1, head_dim)
        k_cache = torch.randn(1, 4, num_kv_heads, 16, head_dim)
        v_cache = torch.randn(1, 4, num_kv_heads, 16, head_dim)

        block_tables = torch.tensor([[0, 1]], dtype=torch.long)
        seq_lens = torch.tensor([0])

        output = paged_attention_forward(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=num_kv_heads,
        )
        assert output.shape == (batch_size, num_heads, 1, head_dim)
        assert torch.zeros_like(output).equal(output)

    def test_out_of_range_block_ids(self):
        """Edge case: out-of-range block IDs should not crash."""
        batch_size = 1
        num_heads = 4
        head_dim = 16
        num_kv_heads = 2

        q = torch.randn(batch_size, num_heads, 1, head_dim)
        k_cache = torch.randn(1, 4, num_kv_heads, 16, head_dim)
        v_cache = torch.randn(1, 4, num_kv_heads, 16, head_dim)

        # Block ID 99 is out of range (>= num_blocks=4)
        block_tables = torch.tensor([[99, 100]], dtype=torch.long)
        seq_lens = torch.tensor([20])

        output = paged_attention_forward(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=num_kv_heads,
        )
        assert output.shape == (batch_size, num_heads, 1, head_dim)
        assert torch.zeros_like(output).equal(output)


# --- Numerical equivalence with plain attention (shape checks cannot see bugs) ---


def _causal_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Plain scaled-dot-product attention for a single sequence, causal.

    Inputs are ``[1, H, T, D]`` (q) and ``[1, T, H, D]`` (k, v); the return is
    ``[1, H, T, D]``. This is the semantics the paged kernel must reproduce.
    """
    head_dim = q.shape[-1]
    scores = torch.einsum("bhtd,bshd->bhts", q, k) * head_dim**-0.5
    seq_len = k.shape[1]
    causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    return torch.einsum("bhts,bshd->bhtd", scores.softmax(dim=-1), v)


def _build_cached(k, v, block_size, n_blocks, physical_order):
    """Place a single sequence's K/V into paged blocks; return (kcache, vcache).

    ``physical_order[logical_block]`` is the physical block id that logical
    block occupies, so a scrambled order exercises non-contiguous tables.
    Cache layout is ``[num_blocks, num_kv_heads, block_size, head_dim]``.
    """
    num_heads = k.shape[2]
    head_dim = k.shape[3]
    seq_len = k.shape[1]
    k_cache = torch.zeros(n_blocks, num_heads, block_size, head_dim)
    v_cache = torch.zeros(n_blocks, num_heads, block_size, head_dim)
    for token in range(seq_len):
        logical, offset = divmod(token, block_size)
        physical = int(physical_order[logical])
        k_cache[physical, :, offset] = k[0, token]
        v_cache[physical, :, offset] = v[0, token]
    return k_cache, v_cache


def test_paged_attention_matches_plain_attention_on_prefill():
    """Prefill through paged blocks must equal plain causal attention exactly.

    Every existing test in this class asserts the output SHAPE, so a kernel
    that attended to the wrong keys would still pass. This compares against a
    reference implementation instead.
    """
    torch.manual_seed(0)
    heads, head_dim, block_size, seq_len = 2, 8, 4, 6
    n_blocks = (seq_len + block_size - 1) // block_size

    q = torch.randn(1, heads, seq_len, head_dim)
    k = torch.randn(1, seq_len, heads, head_dim)
    v = torch.randn(1, seq_len, heads, head_dim)
    expected = _causal_reference(q, k, v)

    k_cache, v_cache = _build_cached(k, v, block_size, n_blocks, list(range(n_blocks)))
    out = paged_attention_forward(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=torch.arange(n_blocks).unsqueeze(0),
        seq_lens=torch.tensor([seq_len]),
        num_kv_heads=heads,
        block_size=block_size,
        query_lens=torch.tensor([seq_len]),
    )
    assert torch.allclose(out, expected, atol=1e-6), (
        f"paged prefill diverged from plain attention: max|delta|={(out - expected).abs().max().item():.2e}"
    )


def test_paged_attention_matches_plain_attention_on_decode():
    """A single decode query must attend to the WHOLE cached context.

    The causal overlay is only meaningful for multi-query prefill rows; applying
    it to a decode row would mask everything past key 0 and collapse the
    attention onto the first token. Comparing against a plain full-context
    reference catches that.
    """
    torch.manual_seed(0)
    heads, head_dim, block_size, seq_len = 2, 8, 4, 6
    n_blocks = (seq_len + block_size - 1) // block_size

    q = torch.randn(1, heads, 1, head_dim)
    k = torch.randn(1, seq_len, heads, head_dim)
    v = torch.randn(1, seq_len, heads, head_dim)

    # Reference: no causal mask needed, the query is the last position.
    scores = torch.einsum("bhtd,bshd->bhts", q, k) * head_dim**-0.5
    expected = torch.einsum("bhts,bshd->bhtd", scores.softmax(dim=-1), v)

    k_cache, v_cache = _build_cached(k, v, block_size, n_blocks, list(range(n_blocks)))
    out = paged_attention_forward(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=torch.arange(n_blocks).unsqueeze(0),
        seq_lens=torch.tensor([seq_len]),
        num_kv_heads=heads,
        block_size=block_size,
        query_lens=torch.tensor([1]),
    )
    assert torch.allclose(out, expected, atol=1e-6), (
        f"paged decode diverged from plain attention: max|delta|={(out - expected).abs().max().item():.2e}"
    )


def test_paged_attention_handles_scrambled_blocks_and_partial_last_block():
    """Non-contiguous physical blocks and a partly-filled tail must be exact.

    ``seq_len=10`` with ``block_size=4`` leaves the last block half full, and
    the block table is deliberately out of order — the two cases most likely to
    break a hand-rolled gather.
    """
    torch.manual_seed(0)
    heads, head_dim, block_size, seq_len = 2, 8, 4, 10
    n_blocks = (seq_len + block_size - 1) // block_size

    q = torch.randn(1, heads, seq_len, head_dim)
    k = torch.randn(1, seq_len, heads, head_dim)
    v = torch.randn(1, seq_len, heads, head_dim)
    expected = _causal_reference(q, k, v)

    physical_order = [2, 0, 1]  # logical block i lives in physical block physical_order[i]
    k_cache, v_cache = _build_cached(k, v, block_size, n_blocks, physical_order)
    out = paged_attention_forward(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=torch.tensor([physical_order]),
        seq_lens=torch.tensor([seq_len]),
        num_kv_heads=heads,
        block_size=block_size,
        query_lens=torch.tensor([seq_len]),
    )
    assert torch.allclose(out, expected, atol=1e-6), (
        f"scrambled-block paged attention diverged: max|delta|={(out - expected).abs().max().item():.2e}"
    )


class TestForkReferenceCounting:
    """A forked pair must be safe to free in EITHER order.

    ``fork_sequence`` shares blocks by incrementing every allocator's refcount, so
    freeing one sequence must decrement but not release them while the other still
    reads them. The existing fork tests only assert the returned table and token
    count — nothing ever freed a forked pair, so a missing ``fork`` increment (or a
    premature release) would not have shown up.
    """

    def test_freeing_the_source_leaves_the_fork_usable(self):
        from llm.core.paged_attention.block_manager import BlockManager

        manager = BlockManager(num_blocks=16, block_size=8, num_layers=1)
        initial_free = manager.num_free_blocks

        source_blocks = manager.allocate_sequence(seq_id=1, num_tokens=16)
        fork_blocks = manager.fork_sequence(src_seq_id=1, dst_seq_id=2)
        assert fork_blocks == source_blocks, "a fork must share the same physical blocks"
        assert manager.num_free_blocks == initial_free - len(source_blocks), "forking allocated new blocks"

        manager.free_sequence(1)

        # The FORK still owns them, so nothing may return to the pool yet.
        assert manager.num_free_blocks == initial_free - len(source_blocks), (
            "freeing the source released blocks the fork still references"
        )
        assert manager.get_block_table(2) == fork_blocks
        assert manager.get_num_tokens(2) == 16

        manager.free_sequence(2)
        assert manager.num_free_blocks == initial_free, "the pair did not fully release its blocks"

    def test_freeing_the_fork_leaves_the_source_usable(self):
        from llm.core.paged_attention.block_manager import BlockManager

        manager = BlockManager(num_blocks=16, block_size=8, num_layers=1)
        initial_free = manager.num_free_blocks

        source_blocks = manager.allocate_sequence(seq_id=1, num_tokens=16)
        manager.fork_sequence(src_seq_id=1, dst_seq_id=2)

        manager.free_sequence(2)

        assert manager.num_free_blocks == initial_free - len(source_blocks), (
            "freeing the fork released blocks the source still references"
        )
        assert manager.get_block_table(1) == source_blocks

        manager.free_sequence(1)
        assert manager.num_free_blocks == initial_free

    def test_copy_on_write_hands_the_copier_a_private_block(self):
        """COW decrements the shared count and the caller remaps its table.

        ``cow_block`` deliberately does NOT remap (its docstring says the caller
        does), so this asserts both halves: the fresh id differs, the original
        stops being shared, and after the caller's remap both frees balance.
        """
        from llm.core.paged_attention.block_manager import BlockManager

        manager = BlockManager(num_blocks=16, block_size=8, num_layers=1)
        initial_free = manager.num_free_blocks

        source_blocks = manager.allocate_sequence(seq_id=1, num_tokens=16)
        manager.fork_sequence(src_seq_id=1, dst_seq_id=2)
        assert manager.is_block_shared(source_blocks[0])

        private = manager.cow_block(source_blocks[0])

        assert private != source_blocks[0], "COW returned the shared block itself"
        assert not manager.is_block_shared(source_blocks[0]), (
            "after COW the original should have a single remaining reader (the fork)"
        )
        assert manager.is_block_shared(private) is False, "the private copy must not be shared"

        manager.get_block_table(1)[0] = private
        manager.free_sequence(1)
        manager.free_sequence(2)
        assert manager.num_free_blocks == initial_free
