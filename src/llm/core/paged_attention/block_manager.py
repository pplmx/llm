"""
Block Manager for Paged Attention.

Manages logical-to-physical block mapping for sequences.
"""

from dataclasses import dataclass, field

from llm.core.paged_attention.block_allocator import BlockAllocator


@dataclass
class SequenceBlockInfo:
    """Block information for a single sequence."""

    seq_id: int
    block_table: list[int] = field(default_factory=list)  # List of physical block IDs
    num_tokens: int = 0  # Total tokens in this sequence


class BlockManager:
    """
    Manages block allocation for multiple sequences.

    Maintains a mapping from sequence IDs to their block tables,
    handling allocation, extension, and freeing of blocks.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_layers: int = 1,
    ):
        """
        Initialize the block manager.

        Args:
            num_blocks: Total physical blocks per layer.
            block_size: Tokens per block.
            num_layers: Number of transformer layers.
        """
        self.block_size = block_size
        self.num_layers = num_layers

        # One allocator per layer for independent management
        self.allocators = [BlockAllocator(num_blocks, block_size) for _ in range(num_layers)]

        # Sequence block tables: seq_id -> SequenceBlockInfo
        self.sequences: dict[int, SequenceBlockInfo] = {}

    @property
    def num_free_blocks(self) -> int:
        """Minimum free blocks across all layers."""
        return min(alloc.num_free_blocks for alloc in self.allocators)

    def can_allocate_sequence(self, num_tokens: int) -> bool:
        """
        Check if a new sequence with given tokens can be allocated.

        Args:
            num_tokens: Number of tokens in the sequence.

        Returns:
            True if allocation is possible.
        """
        num_blocks_needed = self._tokens_to_blocks(num_tokens)
        return all(alloc.can_allocate(num_blocks_needed) for alloc in self.allocators)

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> list[int]:
        """
        Allocate blocks for a new sequence.

        Args:
            seq_id: Unique sequence identifier.
            num_tokens: Initial number of tokens.

        Returns:
            List of physical block IDs for the first layer
            (all layers get the same logical structure).

        Raises:
            RuntimeError: If allocation fails.
            ValueError: If sequence already exists.
        """
        if seq_id in self.sequences:
            raise ValueError(f"Sequence {seq_id} already exists")

        num_blocks = self._tokens_to_blocks(num_tokens)

        # Allocate blocks for each layer
        block_tables_per_layer = []
        try:
            for allocator in self.allocators:
                blocks = allocator.allocate_n(num_blocks)
                block_tables_per_layer.append(blocks)
        except RuntimeError:
            # Rollback on failure
            for i, blocks in enumerate(block_tables_per_layer):
                self.allocators[i].free_all(blocks)
            raise

        # Store first layer's block table (representative)
        first_layer_blocks = block_tables_per_layer[0]
        self.sequences[seq_id] = SequenceBlockInfo(
            seq_id=seq_id,
            block_table=first_layer_blocks.copy(),
            num_tokens=num_tokens,
        )

        return first_layer_blocks

    def extend_sequence(self, seq_id: int, num_new_tokens: int) -> list[int]:
        """
        Extend a sequence with additional tokens.

        Allocates new blocks if needed.

        Args:
            seq_id: Sequence to extend.
            num_new_tokens: Number of new tokens to add.

        Returns:
            Updated block table for the sequence.
        """
        if seq_id not in self.sequences:
            raise ValueError(f"Sequence {seq_id} does not exist")

        info = self.sequences[seq_id]
        old_num_blocks = len(info.block_table)
        new_total_tokens = info.num_tokens + num_new_tokens
        new_num_blocks = self._tokens_to_blocks(new_total_tokens)

        blocks_to_add = new_num_blocks - old_num_blocks

        if blocks_to_add > 0:
            # Allocate additional blocks
            for i, allocator in enumerate(self.allocators):
                new_blocks = allocator.allocate_n(blocks_to_add)
                if i == 0:
                    info.block_table.extend(new_blocks)

        info.num_tokens = new_total_tokens
        return info.block_table

    def free_sequence(self, seq_id: int) -> None:
        """
        Free all blocks associated with a sequence.

        Args:
            seq_id: Sequence to free.
        """
        if seq_id not in self.sequences:
            return  # Already freed or never existed

        info = self.sequences.pop(seq_id)

        # Free blocks in all layers
        for allocator in self.allocators:
            allocator.free_all(info.block_table)

    def fork_sequence(self, src_seq_id: int, dst_seq_id: int) -> list[int]:
        """
        Fork a sequence using copy-on-write.

        Creates a new sequence that shares blocks with the source
        until either is modified.

        Args:
            src_seq_id: Source sequence to fork from.
            dst_seq_id: New sequence ID.

        Returns:
            Block table for the new sequence.
        """
        if src_seq_id not in self.sequences:
            raise ValueError(f"Source sequence {src_seq_id} does not exist")
        if dst_seq_id in self.sequences:
            raise ValueError(f"Destination sequence {dst_seq_id} already exists")

        src_info = self.sequences[src_seq_id]

        # Fork blocks (increment ref counts)
        for allocator in self.allocators:
            for block_id in src_info.block_table:
                allocator.fork(block_id)

        # Create new sequence with shared blocks
        self.sequences[dst_seq_id] = SequenceBlockInfo(
            seq_id=dst_seq_id,
            block_table=src_info.block_table.copy(),
            num_tokens=src_info.num_tokens,
        )

        return src_info.block_table.copy()

    def get_block_table(self, seq_id: int) -> list[int]:
        """Get the block table for a sequence."""
        if seq_id not in self.sequences:
            raise ValueError(f"Sequence {seq_id} does not exist")
        return self.sequences[seq_id].block_table

    def get_num_tokens(self, seq_id: int) -> int:
        """Get the number of tokens in a sequence."""
        if seq_id not in self.sequences:
            raise ValueError(f"Sequence {seq_id} does not exist")
        return self.sequences[seq_id].num_tokens

    def _tokens_to_blocks(self, num_tokens: int) -> int:
        """Calculate number of blocks needed for given tokens."""
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def get_all_sequence_ids(self) -> list[int]:
        """Get all active sequence IDs."""
        return list(self.sequences.keys())

    def reset(self) -> None:
        """Reset manager to initial state."""
        self.sequences.clear()
        for allocator in self.allocators:
            allocator.reset()

    def __repr__(self) -> str:
        return (
            f"BlockManager(block_size={self.block_size}, "
            f"num_layers={self.num_layers}, "
            f"free_blocks={self.num_free_blocks}, "
            f"sequences={len(self.sequences)})"
        )
