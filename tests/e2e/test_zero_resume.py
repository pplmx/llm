"""ZeRO Stage-1 multi-rank checkpoint resume.

ZeRO shards the optimizer state across ranks: each rank owns a disjoint slice
of the parameters' moments, so persisting rank 0's ``state_dict`` alone loses
every other rank's state. Before this test, a multi-rank ZeRO run could save a
checkpoint it could not resume — the loader got rank 0's shard on every rank
and refused it (``state dict is for rank 0, this rank is 1``), or, had that tag
check been absent, silently restarted most of the moments from cold while
looking like a successful resume.

The tests drive the REAL ``TrainingEngine.run()`` with
``parallel_strategy="zero"`` on two ranks, then resume from the written
checkpoint and assert:

* the resume starts at the saved epoch (the checkpoint was found and read);
* both ranks come back bit-identical (the sharded all-gather still holds);
* the on-disk optimizer slot carries a shard for EVERY rank, and each shard
  covers exactly that rank's round-robin parameters (the actual fix — without
  the gap the payload held only ``{"0": ...}``).

Two transports are covered, because they exercise different code: CPU/gloo
(always runs, including GPU-less CI) and CUDA/NCCL (``need_gpu(2)``). The NCCL
variant is the one that proves the shard gather survives a real GPU backend —
``all_gather_object`` pickles through a device tensor pair there, and the
engine's device selection maps each rank to its own card.

Marked ``heavy``: each case spawns two training processes and runs two full
engine runs (the same classification the sibling ZeRO and pipeline-parallel
suites use).
"""

from __future__ import annotations

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.support.devices import all_gpu_devices
from tests.support.distributed import make_store, rendezvous_store

pytestmark = pytest.mark.heavy

JOIN_TIMEOUT_S = 300
# Bounded per-collective timeout for the spawn workers' hand-rolled process
# groups (they call ``init_process_group`` directly rather than going through
# ``DistributedManager``, so they do NOT inherit
# ``DistributedConfig.collective_timeout_seconds``).
#
# Without it torch's default is 30 MINUTES: a rank that dies mid-test leaves its
# siblings blocking in a collective far past the 300s parent join, so the parent
# gives up first and reports "ZeRO spawn exceeded 300s" with no cause.
# ``timeout=`` bounds the initial RENDEZVOUS too, hence well above a cold
# ``mp.spawn`` and still under the parent join so the guard speaks first.
PG_COLLECTIVE_TIMEOUT_S = timedelta(seconds=240)
WORLD_SIZE = 2
TPS = 0.01
# A usable card must have real headroom: the box hosts other workloads, and a
# 2-rank run that OOMs reports as an infrastructure flake rather than a defect.
MIN_FREE_BYTES = 8 * 1024**3


def _build_config(checkpoint_dir: Path, *, resume_from: str | None, backend: str) -> Any:
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
        distributed=DistributedConfig(backend=backend, parallel_strategy="zero"),
        checkpoint=CheckpointConfig(
            checkpoint_dir=str(checkpoint_dir),
            resume_from_checkpoint=resume_from,
            save_best=False,
            save_interval=1,
        ),
    )


def _worker(
    rank: int,
    world_size: int,
    host: str,
    port: int,
    checkpoint_dir: str,
    resume: bool,
    backend: str,
    device_indices: list[int | None],
    results: dict,
) -> None:
    try:
        # gloo only transports CPU tensors, so the CPU leg hides every GPU from
        # the children (a forked child can also inherit a live CUDA context).
        # The NCCL leg must instead see the cards and bind this rank to its own
        # before the rendezvous — NCCL attaches to the current device, so
        # spawned ranks that stay on the default would all land on GPU 0.
        if backend == "gloo":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            # ``mp.spawn`` hands every child the same args, so the per-rank
            # card is selected here rather than by the parent.
            torch.cuda.set_device(device_indices[rank])
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        # Join the parent's store: it owns the port from allocation onward, so
        # there is no free window another process could take. See
        # tests/support/distributed.py for the measured failure this avoids.
        store = make_store(host, port, world_size, timeout=PG_COLLECTIVE_TIMEOUT_S)
        dist.init_process_group(backend, store=store, rank=rank, world_size=world_size, timeout=PG_COLLECTIVE_TIMEOUT_S)

        from llm.training.core.engine import TrainingEngine
        from llm.training.distributed.zero import ZeroOptimizer
        from llm.training.tasks.lm_task import LanguageModelingTask
        from tests.support.data import DummyLMDataModule

        torch.manual_seed(0)
        ckpt_dir = Path(checkpoint_dir)
        config = _build_config(
            ckpt_dir,
            resume_from=str(ckpt_dir / "epoch_1") if resume else None,
            backend=backend,
        )
        data_module = DummyLMDataModule(config)
        task = LanguageModelingTask(config, data_module)
        engine = TrainingEngine(config, task, rank=rank, world_size=world_size, data_module=data_module)

        assert isinstance(engine.optimizer, ZeroOptimizer)
        assert engine.is_training_zero
        if backend == "nccl":
            # The NCCL leg is only meaningful if the ranks really landed on
            # distinct cards; two ranks sharing one card would still pass the
            # numeric assertions while proving nothing about the gather.
            assert engine.device.type == "cuda", f"rank {rank} fell back to {engine.device}"

        engine.run()
        results[rank] = {
            "start_epoch": engine.start_epoch,
            "params": [p.detach().clone().cpu() for p in engine.model.parameters()],
            "device": str(engine.device),
            "owned": engine.optimizer.owned_count(),
            "total": len(list(engine.model.parameters())),
            "error": None,
        }
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def _run(checkpoint_dir: Path, *, resume: bool, backend: str = "gloo") -> dict:
    device_indices: list[int | None] = [None] * WORLD_SIZE
    if backend == "nccl":
        gpus = all_gpu_devices(min_free_bytes=MIN_FREE_BYTES)
        if len(gpus) < WORLD_SIZE:
            pytest.skip(f"need {WORLD_SIZE} GPUs with >= {MIN_FREE_BYTES >> 30} GiB free, got {len(gpus)}")
        device_indices = [device.index for device in gpus[:WORLD_SIZE]]

    manager = mp.Manager()
    results: dict = manager.dict()
    # The store must outlive the children: it owns the port they connect to, so
    # closing it before they rendezvous makes every worker block until its own
    # socket timeout. (Closing it right after ``mp.spawn(join=False)`` returns
    # does exactly that -- spawn returns as soon as the processes are forked.)
    with rendezvous_store(WORLD_SIZE, timeout=PG_COLLECTIVE_TIMEOUT_S) as (host, port):
        context = mp.spawn(
            _worker,
            args=(WORLD_SIZE, host, port, str(checkpoint_dir), resume, backend, device_indices, results),
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
                raise TimeoutError(f"ZeRO spawn ({backend}) exceeded {JOIN_TIMEOUT_S}s")
            if context.join(timeout=remaining):
                break
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in dict(results).items()}
    for rank, payload in out.items():
        assert payload.get("error") is None, f"rank {rank}: {payload.get('error')}"
    return out


def _assert_every_rank_shard_on_disk(checkpoint_dir: Path) -> None:
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


def _assert_multi_rank_resume(checkpoint_dir: Path, backend: str) -> None:
    first = _run(checkpoint_dir, resume=False, backend=backend)
    for rank, payload in first.items():
        assert payload["total"] == first[0]["total"]
        # A genuine shard: non-empty but strictly smaller than the full set.
        assert 0 < payload["owned"] < payload["total"], f"rank {rank} owned {payload['owned']}/{payload['total']}"
    assert first[0]["owned"] + first[1]["owned"] == first[0]["total"]

    second = _run(checkpoint_dir, resume=True, backend=backend)
    for rank, payload in second.items():
        # The resume found the checkpoint AND restored the epoch bookkeeping;
        # without the fix the loader raised before training could start.
        assert payload["start_epoch"] == 1, f"rank {rank} start_epoch={payload['start_epoch']}"
    for param_0, param_1 in zip(second[0]["params"], second[1]["params"], strict=True):
        assert torch.equal(param_0, param_1), "ZeRO ranks drifted out of lockstep after resume"


# --- CPU / gloo (always runs, GPU-less CI included) -------------------------


def test_zero_checkpoint_persists_every_rank_shard(tmp_path):
    """The optimizer slot must hold one shard per rank, not rank 0's shards."""
    checkpoint_dir = tmp_path / "ckpt"
    _run(checkpoint_dir, resume=False)
    _assert_every_rank_shard_on_disk(checkpoint_dir)


def test_zero_multi_rank_resume_restores_every_shard(tmp_path):
    """A 2-rank ZeRO run resumes from its own checkpoint, ranks in lockstep."""
    _assert_multi_rank_resume(tmp_path / "ckpt", "gloo")


# --- CUDA / NCCL (the real-GPU validation the guide used to defer) ----------


@pytest.mark.need_gpu(2)
@pytest.mark.slow
def test_zero_checkpoint_persists_every_rank_shard_two_gpu(tmp_path):
    """The shard gather survives a real NCCL backend, not just gloo's pickling."""
    checkpoint_dir = tmp_path / "ckpt"
    _run(checkpoint_dir, resume=False, backend="nccl")
    _assert_every_rank_shard_on_disk(checkpoint_dir)


@pytest.mark.need_gpu(2)
@pytest.mark.slow
def test_zero_multi_rank_resume_restores_every_shard_two_gpu(tmp_path):
    """Multi-rank ZeRO resume works over NCCL, with both ranks on their own GPU."""
    _assert_multi_rank_resume(tmp_path / "ckpt", "nccl")


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


# --- ZeRO Stage-2 on a real GPU backend (ADR-014) --------------------------
#
# Stage-2 shards gradients, so its reduction runs a different collective path
# than Stage-1's `allreduce_zero_grads`. The CPU/gloo tests in
# `tests/training/distributed/test_zero_optimizer.py` cover the semantics; these
# prove the same code produces the same numbers on NCCL, where the collective
# actually crosses devices.
#
# The reference is computed ON THE SAME DEVICE. That detail is the whole point:
# a first probe compared the GPU result against a CPU full-AdamW reference and
# saw max|d| = 0.227, which looks like a bug until you also run a full GPU
# AdamW — it lands at the SAME 0.227, while GPU-Stage-2 vs GPU-full is bit-exact
# at 0.0. CPU-vs-GPU float differences, not ZeRO.


def _stage2_gpu_worker(rank: int, world_size: int, ctx: dict, results: dict) -> None:
    try:
        torch.cuda.set_device(ctx["devices"][rank])
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        store = make_store(ctx["host"], ctx["port"], world_size, timeout=PG_COLLECTIVE_TIMEOUT_S)
        dist.init_process_group("nccl", store=store, rank=rank, world_size=world_size, timeout=PG_COLLECTIVE_TIMEOUT_S)

        from llm.training.distributed.zero import ZeroStage2Optimizer

        def build():
            torch.manual_seed(123)
            model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3)).to(
                torch.device(f"cuda:{ctx['devices'][rank]}")
            )
            torch.manual_seed(999)
            data = torch.randn(6, 4, device=model[0].weight.device)
            return model, data

        model, data = build()
        params = list(model.parameters())
        optimizer = ZeroStage2Optimizer(
            lambda owned: torch.optim.AdamW(owned, lr=0.01),
            params,
            rank=rank,
            world_size=world_size,
        )
        for _ in range(12):
            optimizer.zero_grad()
            model(data).square().mean().backward()
            optimizer.clip_gradients_globally(0.0)
            optimizer.reduce_gradients_to_owners()
            optimizer.step()

        results[rank] = {
            "params": [p.detach().cpu().clone() for p in params],
            "device": str(next(model.parameters()).device),
            "error": None,
        }
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def _stage2_gpu_reference(device_index: int) -> list[torch.Tensor]:
    """Full AdamW on ONE GPU — the reference Stage-2 must match bit-exactly."""
    device = torch.device(f"cuda:{device_index}")
    torch.manual_seed(123)
    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3)).to(device)
    torch.manual_seed(999)
    data = torch.randn(6, 4, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(12):
        optimizer.zero_grad()
        model(data).square().mean().backward()
        optimizer.step()
    return [p.detach().cpu().clone() for p in model.parameters()]


def _run_stage2_gpu() -> tuple[dict, list[torch.Tensor]]:
    gpus = all_gpu_devices(min_free_bytes=MIN_FREE_BYTES)
    if len(gpus) < WORLD_SIZE:
        pytest.skip(f"need {WORLD_SIZE} GPUs with >= {MIN_FREE_BYTES >> 30} GiB free, got {len(gpus)}")
    devices = [device.index for device in gpus[:WORLD_SIZE]]

    manager = mp.Manager()
    results: dict = manager.dict()
    # See the note on the ZeRO resume helper: the store owns the port the
    # children connect to, so it has to outlive them.
    with rendezvous_store(WORLD_SIZE, timeout=PG_COLLECTIVE_TIMEOUT_S) as (host, port):
        context = mp.spawn(
            _stage2_gpu_worker,
            args=(WORLD_SIZE, {"devices": devices, "host": host, "port": port}, results),
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
                raise TimeoutError(f"Stage-2 NCCL spawn exceeded {JOIN_TIMEOUT_S}s")
            if context.join(timeout=remaining):
                break
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in dict(results).items()}
    manager.shutdown()
    for rank, payload in out.items():
        assert payload.get("error") is None, f"rank {rank}: {payload.get('error')}"

    return out, _stage2_gpu_reference(devices[0])


@pytest.mark.need_gpu(2)
def test_zero_stage2_matches_full_optimizer_on_two_gpus():
    """Stage-2 over NCCL reaches the same weights as a full AdamW on the SAME GPU.

    Comparing against a CPU reference would be wrong here: a full GPU AdamW
    differs from a CPU one by ~0.22 on this model, so a CPU reference would
    report a large "error" that has nothing to do with ZeRO.
    """
    out, reference = _run_stage2_gpu()

    for rank, payload in out.items():
        assert payload["device"].startswith("cuda"), (
            f"rank {rank} ran on {payload['device']}, not a GPU — an NCCL test that silently "
            "falls back to CPU proves nothing about the device path"
        )
        for i, (got, want) in enumerate(zip(payload["params"], reference, strict=True)):
            assert torch.equal(got, want), (
                f"rank {rank} param {i} not bit-exact to the same-device full-AdamW reference "
                f"(max |d|={(got - want).abs().max().item():.3e})"
            )

    # Both ranks must agree as well (the parameter all-gather still restores a
    # single global state after a sharded step).
    for rank in (1,):
        for i, (got, want) in enumerate(zip(out[rank]["params"], out[0]["params"], strict=True)):
            assert torch.equal(got, want), f"rank {rank} param {i} diverged from rank 0 over NCCL"


@pytest.mark.need_gpu(2)
def test_zero_stage2_drops_non_owned_gradients_on_gpu():
    """The memory contract holds on the device path too: only owned grads remain.

    On NCCL the tensors live in device memory, so this also exercises the
    ``grad = None`` assignment against CUDA tensors rather than CPU ones.
    """
    gpus = all_gpu_devices(min_free_bytes=MIN_FREE_BYTES)
    if len(gpus) < WORLD_SIZE:
        pytest.skip(f"need {WORLD_SIZE} GPUs, got {len(gpus)}")
    devices = [device.index for device in gpus[:WORLD_SIZE]]
    manager = mp.Manager()
    results: dict = manager.dict()
    with rendezvous_store(WORLD_SIZE, timeout=PG_COLLECTIVE_TIMEOUT_S) as (host, port):
        context = mp.spawn(
            _stage2_grad_worker,
            args=(WORLD_SIZE, {"devices": devices, "host": host, "port": port}, results),
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
                raise TimeoutError(f"Stage-2 NCCL grad-shard spawn exceeded {JOIN_TIMEOUT_S}s")
            if context.join(timeout=remaining):
                break
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in dict(results).items()}
    manager.shutdown()

    covered: set[int] = set()
    for rank, payload in out.items():
        assert payload.get("error") is None, f"rank {rank}: {payload.get('error')}"
        assert payload["with_grad"] == payload["owned"], (
            f"rank {rank} holds gradients for {payload['with_grad']} but owns {payload['owned']}"
        )
        covered |= set(payload["owned"])
    assert len(covered) == 4, f"the two shards must cover all 4 params, got {sorted(covered)}"


def _stage2_grad_worker(rank: int, world_size: int, ctx: dict, results: dict) -> None:
    try:
        torch.cuda.set_device(ctx["devices"][rank])
        store = make_store(ctx["host"], ctx["port"], world_size, timeout=PG_COLLECTIVE_TIMEOUT_S)
        dist.init_process_group("nccl", store=store, rank=rank, world_size=world_size, timeout=PG_COLLECTIVE_TIMEOUT_S)

        from llm.training.distributed.zero import ZeroStage2Optimizer

        device = torch.device(f"cuda:{ctx['devices'][rank]}")
        torch.manual_seed(123)
        model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3)).to(device)
        torch.manual_seed(999)
        data = torch.randn(6, 4, device=device)
        params = list(model.parameters())
        optimizer = ZeroStage2Optimizer(
            lambda owned: torch.optim.SGD(owned, lr=0.01), params, rank=rank, world_size=world_size
        )
        optimizer.zero_grad()
        model(data).square().mean().backward()
        optimizer.clip_gradients_globally(0.0)
        optimizer.reduce_gradients_to_owners()

        results[rank] = {
            "with_grad": [i for i, p in enumerate(params) if p.grad is not None],
            "owned": [i for i, owner in enumerate(optimizer._owners) if owner == rank],
            "error": None,
        }
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}
