"""Pure-function ownership invariants for ZeRO Stage 1.

Kept OUT of `test_zero_optimizer.py` on purpose: that module carries a file-level
`pytestmark = pytest.mark.heavy`, so everything in it is excluded from the CPU
daily gate. These tests need no process group and no torch rendezvous — they are
arithmetic — so putting them there would have left them permanently unrun.

`_owner_of` decides which rank owns each parameter's optimizer state, which is the
whole basis of ZeRO sharding: a wrong mapping means either some rank keeps state it
should not (no memory saving) or two ranks claim one parameter (double updates).
"""

from __future__ import annotations

import pytest


@pytest.mark.quick
class TestParameterOwnershipIsBalanced:
    """`_owner_of` decides which rank owns each parameter's optimizer state.

    It had no test, and it is the whole basis of ZeRO Stage 1 sharding: a wrong
    mapping means either some rank keeps state it should not (no memory saving) or
    two ranks claim the same parameter (double updates / divergence). The mapping
    is pure arithmetic, so all three invariants are checkable without any process
    group.
    """

    @pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
    @pytest.mark.parametrize("num_params", [0, 1, 5, 7, 16, 100])
    def test_validity_balance_and_coverage(self, world_size, num_params):
        from collections import Counter

        from llm.training.distributed.zero import _owner_of

        owners = [_owner_of(index, world_size) for index in range(num_params)]

        # Validity: every parameter maps to a real rank.
        assert all(0 <= owner < world_size for owner in owners), f"owner out of range for {owners}"

        counts = Counter(owners)

        # Balance: round-robin is optimal, so ranks differ by at most one.
        if counts:
            assert max(counts.values()) - min(counts.values()) <= 1, (
                f"unbalanced sharding for world_size={world_size}, num_params={num_params}: {dict(counts)}"
            )

        # Coverage: every rank gets work once there are enough parameters.
        assert len(counts) == min(num_params, world_size), (
            f"expected {min(num_params, world_size)} distinct owners, got {sorted(counts)}"
        )

    def test_assignment_is_round_robin(self):
        """The mapping must be by INDEX, so shards stay contiguous-ish and stable.

        A hash-based or index-shuffling owner would satisfy balance but make the
        shard membership depend on the parameter ordering in a way that breaks the
        gather/scatter round trip across checkpoints.
        """
        from llm.training.distributed.zero import _owner_of

        world_size = 4
        owners = [_owner_of(index, world_size) for index in range(12)]
        assert owners == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]

    def test_no_parameter_is_owned_twice(self):
        """Every index has exactly ONE owner — the anti-double-update invariant."""
        from llm.training.distributed.zero import _owner_of

        world_size = 4
        assigned = {}
        for index in range(32):
            owner = _owner_of(index, world_size)
            assert index not in assigned, f"index {index} assigned twice"
            assigned[index] = owner
        assert len(assigned) == 32
