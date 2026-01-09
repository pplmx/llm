"""Tests for Paged Attention components."""

import pytest

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

    def test_reset(self):
        """Test manager reset."""
        manager = BlockManager(num_blocks=100, block_size=16, num_layers=1)

        manager.allocate_sequence(seq_id=1, num_tokens=32)
        manager.allocate_sequence(seq_id=2, num_tokens=48)

        manager.reset()

        assert len(manager.get_all_sequence_ids()) == 0
        assert manager.num_free_blocks == 100
