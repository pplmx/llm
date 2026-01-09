"""
Block Allocator for Paged Attention.

Manages allocation and deallocation of physical memory blocks.
"""

from collections import deque


class BlockAllocator:
    """
    Allocator for physical memory blocks.

    Uses a free-list approach to efficiently manage block allocation
    and deallocation. Supports reference counting for copy-on-write.
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        """
        Initialize the block allocator.

        Args:
            num_blocks: Total number of physical blocks available.
            block_size: Number of tokens per block.
        """
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Free list of block indices
        self.free_blocks: deque[int] = deque(range(num_blocks))

        # Reference counts for copy-on-write
        self.ref_counts: dict[int, int] = {}

    @property
    def num_free_blocks(self) -> int:
        """Number of available blocks."""
        return len(self.free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        """Number of allocated blocks."""
        return self.num_blocks - self.num_free_blocks

    def can_allocate(self, num_blocks: int = 1) -> bool:
        """Check if the requested number of blocks can be allocated."""
        return len(self.free_blocks) >= num_blocks

    def allocate(self) -> int:
        """
        Allocate a single block.

        Returns:
            Block index.

        Raises:
            RuntimeError: If no free blocks are available.
        """
        if not self.free_blocks:
            raise RuntimeError("No free blocks available")

        block_id = self.free_blocks.popleft()
        self.ref_counts[block_id] = 1
        return block_id

    def allocate_n(self, n: int) -> list[int]:
        """
        Allocate multiple blocks.

        Args:
            n: Number of blocks to allocate.

        Returns:
            List of block indices.

        Raises:
            RuntimeError: If not enough free blocks are available.
        """
        if len(self.free_blocks) < n:
            raise RuntimeError(f"Not enough free blocks: requested {n}, available {len(self.free_blocks)}")

        blocks = []
        for _ in range(n):
            blocks.append(self.allocate())
        return blocks

    def free(self, block_id: int) -> None:
        """
        Free a block (decrement reference count).

        The block is only returned to the free list when its
        reference count reaches zero.

        Args:
            block_id: Block index to free.
        """
        if block_id not in self.ref_counts:
            raise ValueError(f"Block {block_id} is not allocated")

        self.ref_counts[block_id] -= 1

        if self.ref_counts[block_id] == 0:
            del self.ref_counts[block_id]
            self.free_blocks.append(block_id)

    def free_all(self, block_ids: list[int]) -> None:
        """Free multiple blocks."""
        for block_id in block_ids:
            self.free(block_id)

    def fork(self, block_id: int) -> int:
        """
        Fork a block for copy-on-write.

        Increments the reference count instead of copying.

        Args:
            block_id: Block to fork.

        Returns:
            Same block_id (sharing the physical block).
        """
        if block_id not in self.ref_counts:
            raise ValueError(f"Block {block_id} is not allocated")

        self.ref_counts[block_id] += 1
        return block_id

    def get_ref_count(self, block_id: int) -> int:
        """Get the reference count of a block."""
        return self.ref_counts.get(block_id, 0)

    def is_shared(self, block_id: int) -> bool:
        """Check if a block is shared (ref_count > 1)."""
        return self.get_ref_count(block_id) > 1

    def copy_on_write(self, block_id: int) -> int:
        """
        Perform copy-on-write if block is shared.

        If the block has ref_count > 1, allocates a new block
        and decrements the old block's ref_count.

        Args:
            block_id: Block that may need copying.

        Returns:
            New block_id (may be same if not shared).
        """
        if not self.is_shared(block_id):
            return block_id

        # Allocate new block
        new_block_id = self.allocate()

        # Decrement old block's ref_count
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            del self.ref_counts[block_id]
            self.free_blocks.append(block_id)

        return new_block_id

    def reset(self) -> None:
        """Reset allocator to initial state."""
        self.free_blocks = deque(range(self.num_blocks))
        self.ref_counts.clear()

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(num_blocks={self.num_blocks}, "
            f"free={self.num_free_blocks}, "
            f"allocated={self.num_allocated_blocks})"
        )
