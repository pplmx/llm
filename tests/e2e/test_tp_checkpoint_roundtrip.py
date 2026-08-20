"""TP checkpoint ON-DISK roundtrip (RIL TASK-205).

The TP milestones (TASK-200 / TASK-202) verified the state-dict gather /
scatter helpers in-memory, but never proved a real TP/2D training run ends
up as a correct, portable on-disk checkpoint through the checkpoint manager.
These tests pin the two guarantees ``llm-serve``/resume rely on:

* **on-disk save == gathered full state dict**: after a real
  ``CheckpointManager`` save on a TP / TP+DP 2D run, ``latest.safetensors``
  must equal ``model_state_dict(model)`` bit-for-bit on every rank;
* **serve-load / resume portability**: the on-disk full state dict loads into
  a fresh single-GPU ``DecoderModel`` (the ``llm-serve`` path) and its forward
  matches the TP model's gathered forward, and re-loading it into the TP model
  (scatter) leaves the forward unchanged.

Skipped automatically when the required GPUs are not available.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path
from string import printable
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llm.data.modules.sft import SFTDataModule
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.core.checkpoint import CheckpointManager
from llm.training.core.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    DistributedConfig,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from llm.training.core.engine import TrainingEngine
from llm.training.core.utils import Logger
from llm.training.distributed import (
    allreduce_dp_grads,
    load_model_state_dict,
    model_state_dict,
    wrap_model_for_training,
)
from llm.training.tasks.sft_task import SFTTask
from tests.support.devices import all_gpu_devices

ROUNDTRIP_MIN_FREE_BYTES = 1 * 1024**3
ROUNDTRIP_JOIN_TIMEOUT_S = 480


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


def _tiny_model(ctx: dict, *, device) -> DecoderModel:
    torch.manual_seed(7)
    return DecoderModel(
        vocab_size=ctx["vocab_size"],
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        max_seq_len=24,
        intermediate_size=64,
        attn_dropout_p=0.0,
        mlp_dropout_p=0.0,
        embedding_dropout_p=0.0,
        qkv_bias=True,
        mlp_bias=True,
        lm_head_bias=True,
        device=device,
    )


def _disk_state(ckpt_dir: Path, stem: str = "latest") -> dict[str, torch.Tensor]:
    """Load a checkpoint sidecar exactly as the loader does (safetensors)."""
    from safetensors.torch import load_file

    return load_file(str(ckpt_dir / f"{stem}.safetensors"))


# ---------------------------------------------------------------------------
# Test 1: 2-GPU pure-TP on-disk roundtrip + serve-load / resume portability
# ---------------------------------------------------------------------------


def _roundtrip_worker(rank: int, world_size: int, ctx: dict, results) -> None:
    try:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dev = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

        model = _tiny_model(ctx, device=dev)
        wrapped = wrap_model_for_training(
            model, parallel_strategy="tp", device=dev, world_size=world_size, tp_size=world_size
        )
        assert wrapped is model

        # A couple of real training steps so the checkpoint is non-trivial.
        # ``allreduce_dp_grads`` (a no-op for the DP dimension here, but it
        # ALSO averages the replicated-param gradients over the TP group) keeps
        # the replicated params bit-identical across ranks — the engine does
        # this every step, and without it the on-disk equality check on ranks
        # != 0 would trip on the FP drift (ISS-254).
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        x = torch.randint(0, ctx["vocab_size"], (2, 12), device=dev, dtype=torch.long)
        for _ in range(2):
            model.train()
            opt.zero_grad()
            loss = model(x).float().mean()
            loss.backward()
            allreduce_dp_grads(model)
            opt.step()

        # Save through the REAL checkpoint manager: rank 0 writes, every rank
        # enters the collective gather (checkpoint manager's TP handling).
        manager = CheckpointManager(
            CheckpointConfig(checkpoint_dir=ctx["ckpt_dir"], save_interval=1, keep_last_n=2, save_best=False),
            rank=rank,
            logger=Logger(rank, LoggingConfig()),
        )
        manager.save_checkpoint(
            epoch=0,
            model=model,
            optimizer=opt,
            scheduler=None,
            scaler=None,
            loss=float(loss.item()),
        )
        # ``save_checkpoint`` now guarantees durable-return (RIL ISS-256):
        # every rank exits the collective save only after rank 0's disk write
        # is complete, so the file below is guaranteed present. (The barrier
        # this comment used to describe was moved INTO save_checkpoint — every
        # caller, including the engine's epoch-end save and PPO's per-batch
        # save, gets the same contract without remembering to add one.)
        dist.barrier()

        # Every rank can read the on-disk file (shared FS); it must equal the
        # in-memory gathered full state dict bit-for-bit.
        disk = _disk_state(Path(ctx["ckpt_dir"]))
        full = model_state_dict(model)  # TP-group collective; every rank enters
        assert set(disk.keys()) == set(full.keys())
        for key in full:
            # Compare on CPU in fp32: the disk tensors are on CPU while the
            # gathered ``full`` is on the CUDA device (assert_close unifies
            # them anyway; doing it explicitly keeps the diff computation clean).
            torch.testing.assert_close(
                disk[key].float().cpu(),
                full[key].float().cpu(),
                atol=0,
                rtol=0,
                msg=f"on-disk checkpoint differs from gathered state on {key}",
            )

        # EVERY rank enters the TP forward (it is a collective over the group);
        # only the local single-rank comparison is rank-0-only. Doing it any
        # other way deadlocks the non-zero ranks who have already left the
        # collective participation.
        model.eval()
        with torch.no_grad():
            tp_logits = model(x)  # TP forward: collective, gathered vocab

        if rank == 0:
            # Serve-load: a FRESH single-GPU model must reproduce the TP model's
            # forward from the on-disk full state dict (llm-serve portability).
            fresh = _tiny_model(ctx, device=dev)
            load_model_state_dict(fresh, disk)
            fresh.eval()
            with torch.no_grad():
                single_logits = fresh(x)  # local single-rank forward
            torch.testing.assert_close(single_logits, tp_logits, atol=1e-5, rtol=1e-5)

        # Resume: re-loading the on-disk dict INTO the TP model (scatter path,
        # a local in-place load, safe on every rank) leaves the forward
        # unchanged — every rank enters the verification forward again.
        load_model_state_dict(model, disk)
        model.eval()
        with torch.no_grad():
            reloaded_logits = model(x)  # collective; every rank enters
        torch.testing.assert_close(reloaded_logits, tp_logits, atol=1e-5, rtol=1e-5)

        results[rank] = {
            "success": True,
            "disksha": hashlib.sha256(json.dumps(sorted(disk.keys())).encode()).hexdigest(),
        }
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_roundtrip(tmp_path: Path, world_size: int) -> None:
    gpu_devices = all_gpu_devices(min_free_bytes=ROUNDTRIP_MIN_FREE_BYTES)
    if len(gpu_devices) < world_size:
        pytest.skip(f"need at least {world_size} free GPUs (TP checkpoint roundtrip)")
    _release_parent_cuda_caches()
    port = _free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    ctx = {"vocab_size": 112, "ckpt_dir": str(tmp_path / "ckpts")}
    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(_roundtrip_worker, args=(world_size, ctx, results), nprocs=world_size, join=False)
    end_at = time.monotonic() + ROUNDTRIP_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"TP checkpoint roundtrip spawn exceeded {ROUNDTRIP_JOIN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break
    for rank in range(world_size):
        assert rank in results, f"rank {rank} produced no result"
        assert results[rank]["success"], f"rank {rank} failed: {results[rank].get('error')}"


@pytest.mark.need_gpu(2)
@pytest.mark.slow
def test_tp_checkpoint_roundtrip_two_gpu(tmp_path):
    """On-disk TP save == gathered state; serve-load + resume reproduce the forward (2 GPUs)."""
    _run_roundtrip(tmp_path, world_size=2)


@pytest.mark.need_gpu(4)
@pytest.mark.slow
def test_tp_checkpoint_roundtrip_four_gpu_deep_tp(tmp_path):
    """Same roundtrip with a deeper TP group (4 GPUs, tp=4 pure TP)."""
    _run_roundtrip(tmp_path, world_size=4)


# ---------------------------------------------------------------------------
# Test 2: 4-GPU TP+DP 2D ENGINE run — the auto-saved on-disk checkpoint must
# equal the gathered full state dict (real training, via the training loop).
# ---------------------------------------------------------------------------


def _build_engine_config(ctx: dict, *, epochs: int = 1, resume_from: str | None = None) -> Config:
    """Tiny-SFT config with REAL labels (max_seq_len 256) + explicit ckpt dir."""
    checkpoint_cfg = CheckpointConfig(checkpoint_dir=ctx["ckpt_dir"], save_interval=1, keep_last_n=2, save_best=False)
    if resume_from is not None:
        checkpoint_cfg.resume_from_checkpoint = resume_from
    return Config(
        model=ModelConfig(
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            vocab_size=ctx["vocab_size"],
            max_seq_len=256,
        ),
        training=TrainingConfig(batch_size=2, epochs=epochs, lr=5e-3, warmup_epochs=0, log_every_n_steps=1),
        data=DataConfig(
            dataset_path=ctx["data_path"],
            max_seq_len=256,
            tokenizer_type="simple",
            tokenizer_path=ctx["tokenizer_path"],
        ),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(backend="nccl", parallel_strategy="tp", tp_size=ctx["tp_size"]),
        checkpoint=checkpoint_cfg,
    )


def _engine_checkpoint_worker(rank: int, world_size: int, device_indices: list[int], ctx: dict, results) -> None:
    try:
        device_index = device_indices[rank]
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device_index)
        torch.manual_seed(42 + rank)

        config = _build_engine_config(ctx)
        data_module = SFTDataModule(config)
        data_module.prepare_data()
        data_module.setup()
        task = SFTTask(config, data_module)
        engine = TrainingEngine(config=config, task=task, rank=rank, world_size=world_size, data_module=data_module)
        engine.run()  # the standard loop auto-saves via CheckpointManager

        # The engine wrote latest.safetensors on rank 0 (collective gather).
        gathered = model_state_dict(engine.model)
        disk = _disk_state(Path(ctx["ckpt_dir"]))
        assert set(disk.keys()) == set(gathered.keys())
        for key in gathered:
            disk_t = disk[key].float().cpu()
            gathered_t = gathered[key].float().cpu()
            torch.testing.assert_close(
                disk_t,
                gathered_t,
                atol=0,
                rtol=0,
                msg=(
                    f"engine on-disk checkpoint differs from gathered state on {key}: "
                    f"max |disk - gathered| = {(disk_t - gathered_t).abs().max().item():.3e} "
                    f"(disk {tuple(disk[key].shape)} vs gathered {tuple(gathered[key].shape)})"
                ),
            )
        assert engine.global_step > 0, "no training ran (vacuous checkpoint check)"
        results[rank] = {"success": True, "steps": engine.global_step}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _prepare_engine_fixtures(tmp_path: Path, ctx: dict) -> None:
    """Write the shared tokenizer + tiny SFT dataset into ``ctx`` (reused by the
    engine-checkpoint equality tests AND the two-stage resume test — the resumed
    stage must train on the SAME data so batch counts and step counts line up)."""
    tokenizer = SimpleCharacterTokenizer([printable])
    tokenizer_path = str(tmp_path / "tokenizer.pt")
    torch.save(tokenizer, tokenizer_path)
    data_path = tmp_path / "sft_data.jsonl"
    with data_path.open("w") as f:
        for i in range(12):
            f.write(json.dumps({"instruction": f"Inst {i}", "input": "", "output": f"Out {i}"}) + "\n")
    ctx["data_path"] = str(data_path)
    ctx["tokenizer_path"] = tokenizer_path
    ctx["vocab_size"] = tokenizer.vocab_size + 10 + (tokenizer.vocab_size % 2)


def _spawn_world(
    world_size: int,
    device_indices: list[int],
    worker: Any,
    worker_args: tuple,
    timeout_s: int,
    what: str,
) -> mp.managers.DictProxy:
    """Spawn ``world_size`` copies of ``worker`` on the given devices and wait
    for them all (kill on timeout); returns the shared results dict."""
    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(worker, args=(world_size, device_indices, *worker_args, results), nprocs=world_size, join=False)
    end_at = time.monotonic() + timeout_s
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"{what} spawn exceeded {timeout_s}s")
        if context.join(timeout=remaining):
            break
    for rank in range(world_size):
        assert rank in results, f"rank {rank} produced no result"
        assert results[rank]["success"], f"rank {rank} failed: {results[rank].get('error')}"
    return results


def _set_master_env(port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def _run_engine_checkpoint(tmp_path: Path, world_size: int, tp_size: int) -> None:
    gpu_devices = all_gpu_devices(min_free_bytes=ROUNDTRIP_MIN_FREE_BYTES)
    if len(gpu_devices) < world_size:
        pytest.skip(f"need at least {world_size} free GPUs (TP+DP 2D engine checkpoint)")
    device_indices = [device.index for device in gpu_devices[:world_size]]
    _release_parent_cuda_caches()

    ctx: dict = {"tp_size": tp_size, "ckpt_dir": str(tmp_path / "ckpts")}
    _prepare_engine_fixtures(tmp_path, ctx)

    _set_master_env(_free_port())
    _spawn_world(
        world_size, device_indices, _engine_checkpoint_worker, (ctx,), ROUNDTRIP_JOIN_TIMEOUT_S, "engine checkpoint"
    )


@pytest.mark.need_gpu(4)
@pytest.mark.slow
def test_engine_tp_dp_2d_on_disk_checkpoint_four_gpu(tmp_path):
    """Real-engine TP+DP 2D auto-saved checkpoint == gathered full state (4 GPUs, tp=2 dp=2).

    Regression-guards ISS-256: the engine's epoch-end save used to let
    non-zero ranks return before rank 0's disk write was durable, so a rank
    reading the checkpoint right after ``run()`` raced the write (seen as
    FileNotFoundError / stale file reads). ``save_checkpoint`` now makes the
    collective save durable-before-return on every rank.
    """
    _run_engine_checkpoint(tmp_path, world_size=4, tp_size=2)


@pytest.mark.need_gpu(6)
@pytest.mark.slow
def test_engine_tp_dp_2d_on_disk_checkpoint_wide_grid_six_gpu(tmp_path):
    """Real-engine TP+DP 2D on-disk checkpoint on a wider DP grid (6 GPUs, tp=2 dp=3)."""
    _run_engine_checkpoint(tmp_path, world_size=6, tp_size=2)


# ---------------------------------------------------------------------------
# Test 3: 4-GPU TP+DP 2D ENGINE resume — a checkpoint saved to disk by one run
# must restart the training loop with the restored optimizer step counter.
# ---------------------------------------------------------------------------


def _engine_resume_stage_one_worker(rank: int, world_size: int, device_indices: list[int], ctx: dict, results) -> None:
    """Stage 1: train one epoch on the 2D grid and save via the engine loop."""
    try:
        device_index = device_indices[rank]
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device_index)
        torch.manual_seed(42 + rank)

        config = _build_engine_config(ctx, epochs=1)
        data_module = SFTDataModule(config)
        data_module.prepare_data()
        data_module.setup()
        task = SFTTask(config, data_module)
        engine = TrainingEngine(config=config, task=task, rank=rank, world_size=world_size, data_module=data_module)
        engine.run()
        if rank == 0:
            assert engine.global_step > 0, "stage-1 run trained no steps (vacuous resume source)"
        results[rank] = {"success": True, "steps": engine.global_step}
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _engine_resume_stage_two_worker(rank: int, world_size: int, device_indices: list[int], ctx: dict, results) -> None:
    """Stage 2: a FRESH engine + process group resumes the on-disk checkpoint and
    must continue training from the restored epoch/step counter."""
    try:
        device_index = device_indices[rank]
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device_index)
        torch.manual_seed(42 + rank)

        # Same 2D grid, same data; resume from the stage-1 epoch checkpoint stem.
        config = _build_engine_config(ctx, epochs=2, resume_from=ctx["resume_path"])
        data_module = SFTDataModule(config)
        data_module.prepare_data()
        data_module.setup()
        task = SFTTask(config, data_module)
        engine = TrainingEngine(config=config, task=task, rank=rank, world_size=world_size, data_module=data_module)
        # The engine's init loads the checkpoint: start_epoch must jump to 1 and
        # the persisted global_step must be restored via extra_state.
        info = {
            "start_epoch": engine.start_epoch,
            "restored_steps": engine.global_step,
        }
        engine.run()
        info["final_steps"] = engine.global_step
        # Assignment of the WHOLE dict (nested mutation of a Manager-stored
        # dict only touches the local copy and is never propagated).
        results[rank] = {"success": True, **info}
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_engine_resume_roundtrip(tmp_path: Path, world_size: int, tp_size: int) -> None:
    gpu_devices = all_gpu_devices(min_free_bytes=ROUNDTRIP_MIN_FREE_BYTES)
    if len(gpu_devices) < world_size:
        pytest.skip(f"need at least {world_size} free GPUs (TP+DP 2D engine resume)")
    device_indices = [device.index for device in gpu_devices[:world_size]]

    ctx: dict = {"tp_size": tp_size, "ckpt_dir": str(tmp_path / "ckpts")}
    _prepare_engine_fixtures(tmp_path, ctx)

    # Stage 1: train 1 epoch, save, then destroy the process group.
    _release_parent_cuda_caches()
    _set_master_env(_free_port())
    stage1 = _spawn_world(
        world_size, device_indices, _engine_resume_stage_one_worker, (ctx,), ROUNDTRIP_JOIN_TIMEOUT_S, "resume stage-1"
    )
    stage1_steps = stage1[0]["steps"]

    # Stage 2: a fresh world resumes from the epoch_1 checkpoint on disk.
    ctx["resume_path"] = str(Path(ctx["ckpt_dir"]) / "epoch_1")
    _release_parent_cuda_caches()
    _set_master_env(_free_port())
    stage2 = _spawn_world(
        world_size, device_indices, _engine_resume_stage_two_worker, (ctx,), ROUNDTRIP_JOIN_TIMEOUT_S, "resume stage-2"
    )
    for rank in range(world_size):
        assert stage2[rank]["start_epoch"] == 1, f"rank {rank}: start_epoch not restored ({stage2[rank]})"
        assert stage2[rank]["restored_steps"] == stage1_steps, (
            f"rank {rank}: global_step not restored ({stage2[rank]['restored_steps']} != {stage1_steps})"
        )
        assert stage2[rank]["final_steps"] == 2 * stage1_steps, (
            f"rank {rank}: resumed run did not continue training ({stage2[rank]['final_steps']} != 2 * {stage1_steps})"
        )


@pytest.mark.need_gpu(4)
@pytest.mark.slow
def test_engine_tp_dp_2d_resume_from_disk_four_gpu(tmp_path):
    """A real-engine TP+DP 2D checkpoint resumes in a fresh world with the
    optimizer step counter / epoch restored, and training continues (TASK-205)."""
    _run_engine_resume_roundtrip(tmp_path, world_size=4, tp_size=2)
