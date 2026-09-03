"""Multi-node launcher rank assignment (RIL TASK-191 / ISS-229).

The documented multi-node flow (``NUM_NODES=2 NODE_RANK=... GPUS_PER_NODE=4``)
was structurally broken: every node spawned ``world_size`` local workers with
ranks ``0..world_size-1`` and never applied ``node_rank``, so both nodes'
workers claimed the same ranks and the rendezvous collided/hung.  These tests
pin the new global-rank arithmetic and the setup wiring.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

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
        # ``select_cuda_index`` reads free VRAM to prefer the fattest GPU
        # (round-138 device select). Simulate "all devices equally free" so
        # the mapping collapses to the historical ``rank % device_count``
        # identity — otherwise the expectation depends on this host's live
        # GPU utilisation (RIL TASK-293/ISS-322).
        patch("llm.training.core.device_select._free_bytes", return_value=10 * 1024**3),
        patch("llm.training.core.distributed.torch.manual_seed"),
    ):
        config = DistributedConfig(num_nodes=2, gpus_per_node=4, node_rank=1)
        manager = DistributedManager(config)
        manager.setup(global_rank=4 + 0, world_size=8, local_rank=0)

    init.assert_called_once_with(
        backend="nccl",
        rank=4,
        world_size=8,
        timeout=timedelta(seconds=config.collective_timeout_seconds),
    )
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
        # See the sibling test: pin free-VRAM readings so device selection is
        # deterministic (equal free -> historical rank mapping), independent
        # of this host's live GPU utilisation.
        patch("llm.training.core.device_select._free_bytes", return_value=10 * 1024**3),
        patch("llm.training.core.distributed.torch.manual_seed"),
    ):
        config = DistributedConfig(num_nodes=1, gpus_per_node=8, node_rank=0)
        manager = DistributedManager(config)
        manager.setup(global_rank=3, world_size=8)

    init.assert_called_once_with(
        backend="nccl",
        rank=3,
        world_size=8,
        timeout=timedelta(seconds=1800),
    )
    set_dev.assert_called_once_with(3)


def test_collective_timeout_default_and_bounds():
    """The group timeout ships conservative by default and is tunable; a
    non-positive value is rejected at config time (RIL TASK-195 / ISS-232)."""
    assert DistributedConfig().collective_timeout_seconds == 1800
    assert DistributedConfig(collective_timeout_seconds=5).collective_timeout_seconds == 5
    with pytest.raises(ValidationError):
        DistributedConfig(collective_timeout_seconds=0)
    with pytest.raises(ValidationError):
        DistributedConfig(collective_timeout_seconds=-3)


def _timeout_worker(rank, world_size, out_path):
    """rank 0 hits a barrier; rank 1 sleeps well past the group timeout."""
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.254"
    os.environ["MASTER_PORT"] = "29650"
    import torch.distributed as dist

    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=2))
        if rank == 0:
            dist.barrier()  # rank 1 never arrives -> bounded TimeoutError
            outcome = "OK(unexpected: barrier completed)"
        else:
            import time

            time.sleep(10)
            outcome = "OK(unexpected: rank1 slept) and barrier"
            dist.barrier()
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        outcome = f"EXC {type(exc).__name__}"
    Path(out_path).write_text(outcome)


def test_single_rank_failure_does_not_hang_forever(tmp_path):
    """A rank that never arrives at a collective must surface as a bounded
    timeout error, not an infinite hang (RIL TASK-195 / ISS-232)."""
    import torch.multiprocessing as mp

    out0 = tmp_path / "r0.txt"
    out1 = tmp_path / "r1.txt"
    procs = [
        mp.Process(target=_timeout_worker, args=(0, 2, str(out0))),
        mp.Process(target=_timeout_worker, args=(1, 2, str(out1))),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
    assert not any(p.is_alive() for p in procs), "workers hung past the timeout"
    r0 = out0.read_text().strip()
    assert r0.startswith("EXC "), f"survivor should raise bounded timeout, got: {r0}"
    assert r0.split()[1] in {"TimeoutError", "RuntimeError"}, f"unexpected exception: {r0}"
