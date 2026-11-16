"""ZeRO Stage-1 multi-rank checkpoint resume.

ZeRO shards the optimizer state across ranks: each rank owns a disjoint slice
of the parameters' moments, so persisting rank 0's ``state_dict`` alone loses
every other rank's state. Before this test, a multi-rank ZeRO run could save a
checkpoint it could not resume — the loader got rank 0's shard on every rank
and refused it (``state dict is for rank 0, this rank is 1``), or, had that tag
check been absent, silently restarted most of the moments from cold while
looking like a successful resume.

The test drives the REAL ``TrainingEngine.run()`` with
``parallel_strategy="zero"`` on two CPU/gloo ranks, then resumes from the
written checkpoint and asserts:

* the resume starts at the saved epoch (the checkpoint was found and read);
* both ranks come back bit-identical (the sharded all-gather still holds);
* the on-disk optimizer slot carries a shard for EVERY rank, and each shard
  covers exactly that rank's round-robin parameters (the actual fix — without
  the gap the payload held only ``{"0": ...}``).

Marked ``heavy``: it spawns two training processes (the same classification the
sibling ZeRO and pipeline-parallel suites use).
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.heavy

JOIN_TIMEOUT_S = 300
WORLD_SIZE = 2
TPS = 0.01


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _build_config(checkpoint_dir: Path, *, resume_from: str | None) -> Any:
    from llm.training.core.config import (
        CheckpointConfig,
        Config,
        DistributedConfig,
        ModelConfig,
        OptimizationConfig,
        TrainingConfig,
    )

    return Config(
        model=ModelConfig(vocab_size=64, hidden_size=16, num_layers=2, num_heads=2, max_seq_len=32),
        # Two epochs of one step each: the run saves a checkpoint per epoch, so
        # the second phase has a concrete epoch to resume from.
        training=TrainingConfig(batch_size=4, epochs=2, num_samples=8),
        optimization=OptimizationConfig(use_amp=False, use_compile=False),
        distributed=DistributedConfig(backend="gloo", parallel_strategy="zero"),
        checkpoint=CheckpointConfig(
            checkpoint_dir=str(checkpoint_dir),
            resume_from_checkpoint=resume_from,
            save_best=False,
            save_interval=1,
        ),
    )


def _worker(rank: int, world_size: int, port: int, checkpoint_dir: str, resume: bool, results: dict) -> None:
    try:
        # gloo only transports CPU tensors, and a forked child can inherit a
        # live CUDA context (mirrors the sibling ZeRO / PP engine e2e).
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        from llm.training.core.engine import TrainingEngine
        from llm.training.distributed.zero import ZeroOptimizer
        from llm.training.tasks.lm_task import LanguageModelingTask
        from tests.support.data import DummyLMDataModule

        torch.manual_seed(0)
        ckpt_dir = Path(checkpoint_dir)
        config = _build_config(ckpt_dir, resume_from=str(ckpt_dir / "epoch_1") if resume else None)
        data_module = DummyLMDataModule(config)
        task = LanguageModelingTask(config, data_module)
        engine = TrainingEngine(config, task, rank=rank, world_size=world_size, data_module=data_module)

        assert isinstance(engine.optimizer, ZeroOptimizer)
        assert engine.is_training_zero

        engine.run()
        results[rank] = {
            "start_epoch": engine.start_epoch,
            "params": [p.detach().clone() for p in engine.model.parameters()],
            "owned": engine.optimizer.owned_count(),
            "total": len(list(engine.model.parameters())),
            "error": None,
        }
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def _run(checkpoint_dir: Path, *, resume: bool) -> dict:
    port = _free_port()
    manager = mp.Manager()
    results: dict = manager.dict()
    context = mp.spawn(
        _worker,
        args=(WORLD_SIZE, port, str(checkpoint_dir), resume, results),
        nprocs=WORLD_SIZE,
        join=False,
    )
    end_at = time.monotonic() + JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"ZeRO resume spawn exceeded {JOIN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in dict(results).items()}
    for rank, payload in out.items():
        assert payload.get("error") is None, f"rank {rank}: {payload.get('error')}"
    return out


def test_zero_checkpoint_persists_every_rank_shard(tmp_path):
    """The optimizer slot must hold one shard per rank, not rank 0's shards."""
    checkpoint_dir = tmp_path / "ckpt"
    _run(checkpoint_dir, resume=False)

    payload = torch.load(checkpoint_dir / "epoch_1.extra_state.pt", weights_only=False)
    optimizer_state = payload["optimizer_state"]
    shards = optimizer_state["__zero_rank_states__"]
    assert sorted(shards) == [str(r) for r in range(WORLD_SIZE)], (
        f"every rank's shard must be persisted, got {sorted(shards)}"
    )
    # Each shard covers exactly its owner's round-robin parameters — a copy of
    # rank 0's shard under every key would satisfy the key set but replay the
    # wrong moments at resume.
    for rank_key, shard in shards.items():
        rank = int(rank_key)
        assert shard["rank"] == rank
        assert shard["world_size"] == WORLD_SIZE
        assert shard["global_indices"], f"rank {rank} shard claims no parameters"
        assert all(index % WORLD_SIZE == rank for index in shard["global_indices"]), (
            f"rank {rank} shard names parameters owned by another rank: {shard['global_indices']}"
        )
        assert shard["inner"]["state"], f"rank {rank} shard captured no optimizer moments"
    assert shards["0"]["global_indices"] != shards["1"]["global_indices"], "shards must be disjoint, not duplicated"

    meta = json.loads((checkpoint_dir / "epoch_1.meta.json").read_text())
    assert meta["epoch"] == 0


def test_zero_multi_rank_resume_restores_every_shard(tmp_path):
    """A 2-rank ZeRO run resumes from its own checkpoint, ranks in lockstep."""
    checkpoint_dir = tmp_path / "ckpt"
    first = _run(checkpoint_dir, resume=False)
    for rank, payload in first.items():
        assert payload["total"] == first[0]["total"]
        # A genuine shard: non-empty but strictly smaller than the full set.
        assert 0 < payload["owned"] < payload["total"], f"rank {rank} owned {payload['owned']}/{payload['total']}"
    assert first[0]["owned"] + first[1]["owned"] == first[0]["total"]

    second = _run(checkpoint_dir, resume=True)
    for rank, payload in second.items():
        # The resume found the checkpoint AND restored the epoch bookkeeping;
        # without the fix the loader raised before training could start.
        assert payload["start_epoch"] == 1, f"rank {rank} start_epoch={payload['start_epoch']}"
    for param_0, param_1 in zip(second[0]["params"], second[1]["params"], strict=True):
        assert torch.equal(param_0, param_1), "ZeRO ranks drifted out of lockstep after resume"


def test_zero_resume_rejects_a_checkpoint_missing_a_rank_shard():
    """A shard set with a hole must fail loud, not restart those moments cold."""
    from llm.training.distributed.zero import ZERO_RANK_STATES_KEY, ZeroOptimizer

    model = torch.nn.Linear(3, 2)
    optimizer = ZeroOptimizer(
        lambda owned: torch.optim.SGD(owned, lr=TPS),
        model.parameters(),
        rank=1,
        world_size=2,
    )
    envelope = {ZERO_RANK_STATES_KEY: {"0": optimizer.state_dict()}}
    with pytest.raises(ValueError, match="no optimizer shard for rank 1"):
        optimizer.load_state_dict(envelope)
