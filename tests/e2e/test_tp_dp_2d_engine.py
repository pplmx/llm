"""TP + data-parallel 2D engine e2e (RIL TASK-202).

Trains a tiny SFT model on ``world_size`` GPUs through the REAL
``TrainingEngine`` with ``parallel_strategy='tp'`` and ``tp_size`` less than
``world_size``, so the DP dimension is live. This exercises the three
integration points the unit/parity tests cannot reach alone:

* engine data sharding per DP group (``data_rank``/``data_world`` from the
  row-major ``[DP][TP]`` layout),
* the step-boundary DP gradient average (``allreduce_dp_grads``) inside the
  standard training loop,
* the full-state-dict checkpoint path (a cross-TP-group collective).

Verification: every rank must complete the same number of steps, and the
full state dict each rank reconstructs must be bit-identical across all
ranks — a shard or replicated-parameter divergence across DP groups (the
failure mode this milestone exists to prevent) fails the equality check.

Skipped automatically when the required GPUs are not available.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import time
from datetime import timedelta
from string import printable
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llm.data.modules.sft import SFTDataModule
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.core.callbacks import Callback
from llm.training.core.config import (
    Config,
    DataConfig,
    DistributedConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from llm.training.core.engine import TrainingEngine
from llm.training.distributed import model_state_dict
from llm.training.tasks.sft_task import SFTTask
from tests.support.devices import all_gpu_devices

ENGINE_2D_MIN_FREE_BYTES = 1 * 1024**3
# Generous but bounded: on a shared box hosting another GPU workload, a
# multi-rank NCCL rendezvous + tiny training run can take minutes to spin up
# (observed 6-GPU engine e2e and 4-GPU wrap timing out at 180-240s under load,
# both passing instantly in isolation). A true deadlock still trips these.
ENGINE_2D_JOIN_TIMEOUT_S = 480
# Bounded per-collective timeout for the spawn workers' process groups (see
# the init_process_group call sites). Mirrors DistributedConfig's
# collective_timeout_seconds for the harness's hand-rolled PGs (RIL TASK-305).
PG_COLLECTIVE_TIMEOUT_S = timedelta(seconds=60)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _release_parent_cuda_caches() -> None:
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(index):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except RuntimeError, torch.AcceleratorError:
            continue


def _build_config(ctx: dict, *, backend: str, parallel_strategy: str, tp_size: int, flash: bool = False) -> Config:
    """Shared tiny-SFT config builder (identical model/loss schedule across runs).

    ``max_seq_len`` must comfortably exceed the verbatim Alpaca-style prompt
    (146 tokens) — at 32/64 the response tokens fall outside the truncation
    window, every label becomes -100, CrossEntropyLoss returns NaN, the SFT
    task's NaN guard returns a constant 0.0 loss, and the run silently trains
    NOTHING (the old e2e was vacuous; surfaced while hard-coding the TP+DP
    milestone).

    ``flash=True`` switches the attention backend to ``flash_attn`` (the
    TASK-204 TP-scope extension) and forces ``use_amp=True``: the flash kernel
    requires half precision, and the engine resolves this to bf16 autocast on
    supported hardware. ``mla=True`` switches to ``MultiLatentAttention`` (the
    TASK-206 TP-scope slice); MLA runs through the sdpa functional in fp32, so
    no AMP is needed. ``moe=True`` enables MoE expert parallelism (TASK-207) on
    the dense MLP blocks (4 experts / 2 ranks, top-2).
    """
    if flash and ctx.get("mla", False):
        raise ValueError("flash and mla are mutually exclusive attention backend overrides")
    attn_impl = "mha"
    if flash:
        attn_impl = "flash_attn"
    elif ctx.get("mla", False):
        attn_impl = "mla"
    model_kwargs: dict[str, Any] = {
        "hidden_size": 32,
        "num_layers": 2,
        "num_heads": 4,  # 32/4 = 8 dim per head; divisible by tp_size
        "vocab_size": ctx["vocab_size"],
        "max_seq_len": 256,
        "attn_impl": attn_impl,
    }
    if ctx.get("moe", False):
        # Expert parallelism shards the expert dimension: num_experts must
        # divide evenly by tp_size (4 / 2 = 2 local experts per rank).
        model_kwargs.update(num_experts=4, top_k=2, mlp_impl="moe")
    return Config(
        model=ModelConfig(**model_kwargs),
        training=TrainingConfig(
            batch_size=2,
            epochs=1,
            lr=5e-3,
            warmup_epochs=0,
            log_every_n_steps=1,
        ),
        data=DataConfig(
            dataset_path=ctx["data_path"],
            max_seq_len=256,
            tokenizer_type="simple",
            tokenizer_path=ctx["tokenizer_path"],
        ),
        optimization=OptimizationConfig(use_compile=False, use_amp=flash, num_workers=0),
        distributed=DistributedConfig(backend=backend, parallel_strategy=parallel_strategy, tp_size=tp_size),
    )


class _CaptureEpochLoss(Callback):
    """Records the last epoch's avg_loss / val_loss from ``on_epoch_end``."""

    def __init__(self) -> None:
        self.avg_loss: float | None = None
        self.val_loss: float | None = None

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs and "avg_loss" in logs:
            self.avg_loss = logs["avg_loss"]
        if logs and "val_loss" in logs:
            self.val_loss = logs["val_loss"]


def _engine_2d_worker(rank: int, world_size: int, device_indices: list[int], ctx: dict, results) -> None:
    try:
        device_index = device_indices[rank]
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            # Bound every collective (RIL TASK-305): the library path uses
            # DistributedConfig.collective_timeout_seconds, but these test
            # spawn workers call init_process_group directly — an unbounded
            # PG left survivors blocking ~forever on the next collective when
            # a sibling rank died mid-training under box flakiness (observed
            # as a blind 480s parent-join timeout AFTER training completed).
            # 60s is far beyond these tiny models' microsecond collectives but
            # turns a dead-sibling stall into a fast torch.distributed
            # TimeoutError with a real message.
            timeout=PG_COLLECTIVE_TIMEOUT_S,
        )
        torch.cuda.set_device(device_index)
        torch.manual_seed(42 + rank)  # the launcher's per-rank seed sequence

        config = _build_config(
            ctx,
            backend="nccl",
            parallel_strategy="tp",
            tp_size=ctx["tp_size"],
            flash=ctx.get("flash", False),
        )
        data_module = SFTDataModule(config)
        data_module.prepare_data()
        data_module.setup()
        task = SFTTask(config, data_module)
        # The process group is already initialised (as the CLI does before
        # constructing the engine); the engine broadcasts initial weights from
        # rank 0 so the per-rank seeds do not diverge the TP shards.
        engine = TrainingEngine(config=config, task=task, rank=rank, world_size=world_size, data_module=data_module)
        # Snapshot BEFORE training so we can prove training actually moved
        # weights (a vacuous no-gradient run keeps the state identical and the
        # cross-rank equality below would trivially pass — regression guard).
        before = model_state_dict(engine.model)
        before_digests = {
            key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for key, value in before.items()
        }
        engine.run()

        steps = engine.global_step
        # Every rank reconstructs the full model (TP-group collective); all
        # ranks must produce the bit-identical dict — no DP divergence.
        full = model_state_dict(engine.model)
        digests = {
            key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for key, value in full.items()
        }
        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, digests)
        for other in gathered:
            if other != digests:
                diff_keys = sorted(k for k in digests if other.get(k) != digests[k])
                raise AssertionError(
                    f"rank {rank}: full state dict diverged from another rank after training; "
                    f"first divergent keys: {diff_keys[:8]}"
                )
        assert steps > 0, f"rank {rank}: no training steps ran"
        # Vacuous-run guard: the optimizer must have MOVED at least one
        # weight, or the SFT NaN-guard silently masked a zero-loss step (the
        # reason the original e2e passed without training).
        moved = [key for key in before_digests if before_digests[key] != digests.get(key)]
        assert moved, f"rank {rank}: no parameter changed across training — the run was a no-op"
        results[rank] = {"success": True, "steps": steps}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_engine_2d(
    tmp_path, world_size: int, tp_size: int, *, flash: bool = False, mla: bool = False, moe: bool = False
) -> None:
    gpu_devices = all_gpu_devices(min_free_bytes=ENGINE_2D_MIN_FREE_BYTES)
    if len(gpu_devices) < world_size:
        pytest.skip(f"need at least {world_size} free GPUs (TP+DP 2D engine e2e)")
    device_indices = [device.index for device in gpu_devices[:world_size]]
    _release_parent_cuda_caches()

    # Tokenizer + corpus (shared read-only across the spawned workers).
    tokenizer = SimpleCharacterTokenizer([printable])
    tokenizer_path = str(tmp_path / "tokenizer.pt")
    torch.save(tokenizer, tokenizer_path)
    data_path = tmp_path / "sft_data.jsonl"
    items = [{"instruction": f"Inst {i}", "input": "", "output": f"Out {i}"} for i in range(12)]
    with data_path.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    port = _free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    ctx = {
        "data_path": str(data_path),
        "tokenizer_path": tokenizer_path,
        # TP requires every partitioned axis to divide evenly by tp_size —
        # force an EVEN vocab so the lm_head / embedding axes partition.
        "vocab_size": tokenizer.vocab_size + 10 + (tokenizer.vocab_size % 2),
        "tp_size": tp_size,
        "flash": flash,
        "mla": mla,
        "moe": moe,
    }
    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(
        _engine_2d_worker,
        args=(world_size, device_indices, ctx, results),
        nprocs=world_size,
        join=False,
    )
    end_at = time.monotonic() + ENGINE_2D_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            alive = [i for i, process in enumerate(context.processes) if process.is_alive()]
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            # Diagnose the stall instead of reporting a blind timeout: the
            # observed failure (RIL TASK-305) was all ranks finishing
            # training, then one dying mid-collective and the survivors
            # blocking until this join fired. Knowing WHO is alive and WHO
            # produced results pinpoints the dead rank.
            reported = [i for i in range(world_size) if i in results]
            raise TimeoutError(
                f"TP+DP 2D engine e2e spawn exceeded {ENGINE_2D_JOIN_TIMEOUT_S}s; "
                f"ranks still alive at timeout: {alive}; ranks with results: {reported}"
            )
        if context.join(timeout=remaining):
            break
    for rank in range(world_size):
        assert rank in results, f"rank {rank} produced no result"
        assert results[rank]["success"], f"rank {rank} failed: {results[rank].get('error')}"
    step_counts = {results[rank]["steps"] for rank in range(world_size)}
    assert len(step_counts) == 1, f"ranks stepped unevenly across DP groups: {step_counts}"


@pytest.mark.need_gpu(4)
@pytest.mark.slow
def test_engine_tp_dp_2d_four_gpu(tmp_path):
    """Real-engine TP+DP 2D training stays in lockstep across 4 GPUs (tp=2 dp=2)."""
    _run_engine_2d(tmp_path, world_size=4, tp_size=2)


@pytest.mark.need_gpu(4)
@pytest.mark.slow
def test_engine_tp_dp_2d_flash_attn_four_gpu(tmp_path):
    """Real-engine TP+DP 2D with the flash-attention backend (TASK-204 TP-scope
    extension): the transformed FlashAttention must train through the REAL loop
    under bf16 AMP autocast and stay bit-identical across DP groups."""
    _run_engine_2d(tmp_path, world_size=4, tp_size=2, flash=True)


@pytest.mark.need_gpu(4)
@pytest.mark.slow
def test_engine_tp_dp_2d_mla_four_gpu(tmp_path):
    """Real-engine TP+DP 2D with the MultiLatentAttention backend (TASK-206,
    TASK-204 mla leg): the block-interleaved K/V column slice must train through
    the REAL loop (fp32, no AMP needed for the sdpa functional) and stay
    bit-identical across DP groups — including the replicated ``latents``
    parameter averaged by ``allreduce_dp_grads``."""
    _run_engine_2d(tmp_path, world_size=4, tp_size=2, mla=True)


@pytest.mark.need_gpu(4)
@pytest.mark.slow
def test_engine_tp_dp_2d_moe_four_gpu(tmp_path):
    """Real-engine TP+DP 2D with MoE expert parallelism on (TASK-207): the EP
    slice (replicated gate + rank-local experts + the step-boundary
    ``allreduce_dp_grads`` expert-skip / gate-SUM) must train through the REAL
    loop with dead-expert-tolerant routing and stay bit-identical across DP
    groups — including the CHECKPOINT gather, where the full expert set is
    rebuilt rank-major from the per-rank shards."""
    _run_engine_2d(tmp_path, world_size=4, tp_size=2, moe=True)


@pytest.mark.need_gpu(4)
@pytest.mark.slow
def test_engine_tp_dp_2d_wide_grid_six_gpu(tmp_path):
    """Real-engine TP+DP 2D with a wider DP grid (6 GPUs, tp=2 dp=3)."""
    _run_engine_2d(tmp_path, world_size=6, tp_size=2)


def _pure_tp_metric_worker(rank: int, world_size: int, ctx: dict, results) -> None:
    """Pure TP (tp_size == world_size == 2) on CPU/gloo: report the epoch loss."""
    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=PG_COLLECTIVE_TIMEOUT_S)
        torch.manual_seed(42 + rank)

        config = _build_config(ctx, backend="gloo", parallel_strategy="tp", tp_size=world_size)
        data_module = SFTDataModule(config)
        data_module.prepare_data()
        data_module.setup()
        task = SFTTask(config, data_module)
        cap = _CaptureEpochLoss()
        engine = TrainingEngine(
            config=config,
            task=task,
            rank=rank,
            world_size=world_size,
            data_module=data_module,
            callbacks=[cap],
        )
        engine.run()
        assert cap.avg_loss is not None, "engine never fired on_epoch_end with avg_loss"
        results[rank] = {"success": True, "avg_loss": cap.avg_loss, "val_loss": cap.val_loss}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _single_rank_avg_loss(ctx: dict) -> float:
    """Reference: the SAME config/data as one plain-rank run (TP is a no-op
    at world_size=1, so this is exactly the value pure-TP must report)."""
    torch.manual_seed(42)
    config = _build_config(ctx, backend="gloo", parallel_strategy="tp", tp_size=2)
    data_module = SFTDataModule(config)
    data_module.prepare_data()
    data_module.setup()
    task = SFTTask(config, data_module)
    cap = _CaptureEpochLoss()
    engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=data_module, callbacks=[cap])
    engine.run()
    assert cap.avg_loss is not None
    return cap.avg_loss


@pytest.mark.quick
def test_pure_tp_reports_true_metric_loss_not_divided_by_world(tmp_path):
    """Regression for RIL ISS-252: pure TP (every rank trains the SAME data)
    must report the true epoch loss, not the single-rank loss / world_size.

    The engine metrics reducer divides by the number of DISTINCT data shards
    (1 under pure TP, world elsewhere). A /world reduce would report ref/2
    here, so this test fails on the pre-fix behaviour.
    """
    tokenizer = SimpleCharacterTokenizer([printable])
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, tokenizer_path)
    data_path = tmp_path / "sft_data.jsonl"
    with data_path.open("w") as f:
        for i in range(12):
            f.write(json.dumps({"instruction": f"Inst {i}", "input": "", "output": f"Out {i}"}) + "\n")
    ctx = {
        "data_path": str(data_path),
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": tokenizer.vocab_size + 10 + (tokenizer.vocab_size % 2),
        "tp_size": 2,
    }

    port = _free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(_pure_tp_metric_worker, args=(2, ctx, results), nprocs=2, join=False)
    end_at = time.monotonic() + ENGINE_2D_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            alive = [i for i, process in enumerate(context.processes) if process.is_alive()]
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(
                f"pure-TP loss spawn exceeded timeout; ranks alive: {alive}; "
                f"ranks with results: {[i for i in range(2) if i in results]}"
            )
        if context.join(timeout=remaining):
            break
    for rank in range(2):
        assert rank in results, f"rank {rank} produced no result"
        assert results[rank]["success"], f"rank {rank} failed: {results[rank].get('error')}"

    # Both pure-TP ranks saw the same data — identical reported loss.
    avg_0, avg_1 = results[0]["avg_loss"], results[1]["avg_loss"]
    assert abs(avg_0 - avg_1) < 1e-6, f"pure-TP ranks disagree on loss: {avg_0} vs {avg_1}"
    # Vacuous-run guard: the loss must be a REAL finite CE value (~log(vocab)),
    # not the 0.0 placeholder the SFT task returns on a NaN step.
    assert math.isfinite(avg_0), f"pure-TP loss {avg_0} is not finite"
    assert avg_0 > 0, f"pure-TP loss {avg_0} is not a real training loss"

    ref = _single_rank_avg_loss(ctx)
    assert avg_0 == pytest.approx(ref, rel=1e-2), (
        f"pure-TP reported epoch loss {avg_0} != single-rank reference {ref} "
        f"(the ISS-252 /world bug would report ~{ref / 2})"
    )
