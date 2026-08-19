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
import os
import socket
import time
from string import printable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llm.data.modules.sft import SFTDataModule
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
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
ENGINE_2D_JOIN_TIMEOUT_S = 240


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


def _engine_2d_worker(rank: int, world_size: int, device_indices: list[int], ctx: dict, results) -> None:
    try:
        device_index = device_indices[rank]
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device_index)
        torch.manual_seed(42 + rank)  # the launcher's per-rank seed sequence

        tp_size = ctx["tp_size"]
        config = Config(
            model=ModelConfig(
                hidden_size=32,
                num_layers=2,
                num_heads=4,  # 32/4 = 8 dim per head; divisible by tp_size
                vocab_size=ctx["vocab_size"],
                max_seq_len=32,
            ),
            training=TrainingConfig(
                batch_size=2,
                epochs=1,
                lr=5e-3,
                warmup_epochs=0,
                log_every_n_steps=1,
            ),
            data=DataConfig(
                dataset_path=ctx["data_path"],
                max_seq_len=32,
                tokenizer_type="simple",
                tokenizer_path=ctx["tokenizer_path"],
            ),
            optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
            distributed=DistributedConfig(backend="nccl", parallel_strategy="tp", tp_size=tp_size),
        )

        data_module = SFTDataModule(config)
        data_module.prepare_data()
        data_module.setup()
        task = SFTTask(config, data_module)
        # The process group is already initialised (as the CLI does before
        # constructing the engine); the engine broadcasts initial weights from
        # rank 0 so the per-rank seeds do not diverge the TP shards.
        engine = TrainingEngine(config=config, task=task, rank=rank, world_size=world_size, data_module=data_module)
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
            assert other == digests, f"rank {rank}: full state dict diverged from another rank after training"
        assert steps > 0, f"rank {rank}: no training steps ran"
        results[rank] = {"success": True, "steps": steps}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_engine_2d(tmp_path, world_size: int, tp_size: int) -> None:
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
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"TP+DP 2D engine e2e spawn exceeded {ENGINE_2D_JOIN_TIMEOUT_S}s")
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
def test_engine_tp_dp_2d_wide_grid_six_gpu(tmp_path):
    """Real-engine TP+DP 2D with a wider DP grid (6 GPUs, tp=2 dp=3)."""
    _run_engine_2d(tmp_path, world_size=6, tp_size=2)
