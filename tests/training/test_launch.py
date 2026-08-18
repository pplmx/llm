"""Multi-node launcher rank assignment (RIL TASK-191 / ISS-229).

The documented multi-node flow (``NUM_NODES=2 NODE_RANK=... GPUS_PER_NODE=4``)
was structurally broken: every node spawned ``world_size`` local workers with
ranks ``0..world_size-1`` and never applied ``node_rank``, so both nodes'
workers claimed the same ranks and the rendezvous collided/hung.  These tests
pin the new global-rank arithmetic and the setup wiring.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llm.training.core.config import DistributedConfig
from llm.training.core.distributed import DistributedManager
from llm.training.train import _global_rank


@pytest.mark.parametrize(
    ("node_rank", "local_rank", "local_world_size", "expected"),
    [
        (0, 3, 4, 3),  # node 0 keeps local ranks
        (1, 0, 4, 4),  # node 1 starts where node 0 ended
        (1, 3, 4, 7),
        (2, 1, 4, 9),
        (0, 0, 1, 0),  # 1 GPU/node
        (2, 0, 1, 2),  # 1 GPU/node, nodes 0..2
        (5, 2, 8, 42),
    ],
)
def test_global_rank_offsets_by_node(node_rank, local_rank, local_world_size, expected):
    assert _global_rank(node_rank, local_rank, local_world_size) == expected


def test_setup_inits_group_with_global_rank_and_device_with_local_rank():
    """``setup`` must rendezvous with the GLOBAL rank and select the CUDA
    device by the LOCAL rank — the two can differ (node 1, local 0 -> global 4)."""
    with (
        patch("llm.training.core.distributed.dist.init_process_group") as init,
        patch("llm.training.core.distributed.torch.cuda.set_device") as set_dev,
        patch("llm.training.core.distributed.torch.cuda.manual_seed_all"),
        patch("llm.training.core.distributed.torch.cuda.is_available", return_value=True),
        patch("llm.training.core.distributed.torch.cuda.device_count", return_value=8),
        patch("llm.training.core.distributed.torch.manual_seed"),
    ):
        config = DistributedConfig(num_nodes=2, gpus_per_node=4, node_rank=1)
        manager = DistributedManager(config)
        manager.setup(global_rank=4 + 0, world_size=8, local_rank=0)

    init.assert_called_once_with(backend="nccl", rank=4, world_size=8)
    set_dev.assert_called_once_with(0)


def test_setup_defaults_local_rank_to_global_rank():
    """Callers (e.g. single-node) that only pass a rank keep the old device
    mapping — global and local coincide so nothing changes."""
    with (
        patch("llm.training.core.distributed.dist.init_process_group") as init,
        patch("llm.training.core.distributed.torch.cuda.set_device") as set_dev,
        patch("llm.training.core.distributed.torch.cuda.manual_seed_all"),
        patch("llm.training.core.distributed.torch.cuda.is_available", return_value=True),
        patch("llm.training.core.distributed.torch.cuda.device_count", return_value=8),
        patch("llm.training.core.distributed.torch.manual_seed"),
    ):
        config = DistributedConfig(num_nodes=1, gpus_per_node=8, node_rank=0)
        manager = DistributedManager(config)
        manager.setup(global_rank=3, world_size=8)

    init.assert_called_once_with(backend="nccl", rank=3, world_size=8)
    set_dev.assert_called_once_with(3)
