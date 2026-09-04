"""Pipeline-parallelism engine e2e (RIL DEC-049 / TASK-210).

Trains a tiny LM model through the REAL ``TrainingEngine`` with
``parallel_strategy='pp'`` on ``world_size`` CPU processes (gloo). This covers
the integration points the unit/parity tests cannot reach alone:

* the engine wiring ``wrap_model_for_training`` -> stage module (``_pp`` tag),
* the PP branch of the standard loop (``_pp_train_step`` via
  ``schedule.step``) driving real forward/backward across ranks,
* the PP-aware global-norm clip and the standard optimizer/scheduler/callback
  machinery around it,
* the full state-dict checkpoint path — a cross-stage collective EVERY rank
  must enter (engine.py gates PP into the save like FSDP/TP), including
  RESUME (each rank scatters its own slice from the global-named dict).

Verification: every rank runs the same number of steps, weights actually
moved (no vacuous zero-loss run), the reconstructed full state dict is
bit-identical across ranks after both the epoch-1 save and the epoch-2
resume-and-continue, and the resumed engine senses ``start_epoch == 1`` and
advances ``global_step`` past the first run. Runs on gloo/CPU with spawn
start-method so it verifies in CI without GPUs — CUDA_VISIBLE_DEVICES is
cleared for the spawned children so the engine picks the CPU path.
"""

from __future__ import annotations

import hashlib
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llm.training.core.checkpoint import load_checkpoint_payload
from llm.training.core.config import (
    Config,
    DistributedConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from llm.training.core.engine import TrainingEngine
from llm.training.distributed import model_state_dict
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.sft_task import SFTTask
from tests.support.data import DummyLMDataModule

PP_ENGINE_JOIN_TIMEOUT_S = 300


def _dict_collate(batch: list) -> dict:
    """Collate TensorDataset rows into an SFT-style dict batch.

    Carries the extra ``attention_mask`` key SFTDataModule emits — the PP stage
    forward must DROP it without changing the loss (RIL ISS-339 / TASK-304).
    """
    input_ids, labels = zip(*batch, strict=True)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
    }


class _SFTDictDataModule(DummyLMDataModule):
    """DummyLMDataModule that collates SFT-style dict batches with a mask key."""

    def train_dataloader(self, rank, world_size, device=None):
        loader, sampler = super().train_dataloader(rank, world_size, device)
        loader.collate_fn = _dict_collate
        return loader, sampler


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _build_config(
    *,
    checkpoint_dir: str,
    epochs: int,
    resume_from_checkpoint: str | None = None,
    pp_size: int | None = None,
    pp_n_microbatches: int | None = None,
    amp_bf16: bool = False,
) -> Config:
    opt_kwargs: dict = {"use_compile": False, "use_amp": amp_bf16, "gradient_accumulation_steps": 2}
    if amp_bf16:
        opt_kwargs["amp_dtype"] = "bfloat16"
    config = Config(
        model=ModelConfig(
            vocab_size=64,
            hidden_size=16,
            num_layers=4,  # 2 stages x 2 blocks
            num_heads=2,
            max_seq_len=32,
        ),
        training=TrainingConfig(batch_size=2, epochs=epochs, num_samples=16, lr=2e-2, log_every_n_steps=1),
        optimization=OptimizationConfig(**opt_kwargs),
        distributed=DistributedConfig(
            backend="gloo",
            parallel_strategy="pp",
            **({"pp_size": pp_size} if pp_size is not None else {}),
            **({"pp_n_microbatches": pp_n_microbatches} if pp_n_microbatches is not None else {}),
        ),
    )
    config.checkpoint.save_interval = 1
    config.checkpoint.checkpoint_dir = checkpoint_dir
    if resume_from_checkpoint is not None:
        config.checkpoint.resume_from_checkpoint = resume_from_checkpoint
    return config


def _digests(model_state: dict) -> dict[str, str]:
    return {
        key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for key, value in model_state.items()
    }


def _first_moment(optimizer_state: dict) -> torch.Tensor | None:
    """The first Adam exp_avg in an optimizer state dict, if the state is non-empty."""
    state = optimizer_state.get("state", {})
    for entry in state.values():
        exp_avg = entry.get("exp_avg")
        if exp_avg is not None:
            return exp_avg.detach().cpu()
    return None


def _assert_ranks_bit_identical(rank: int, digests: dict[str, str], world_size: int, phase: str) -> None:
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, digests)
    for other in gathered:
        if other != digests:
            diff_keys = sorted(k for k in digests if other.get(k) != digests[k])
            raise AssertionError(
                f"rank {rank}: full state dict diverged after {phase}; first divergent keys: {diff_keys[:8]}"
            )


def _worker(rank: int, world_size: int, ctx: dict, results) -> None:
    try:
        # Force the CPU path so the run is deterministic regardless of GPUs.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.manual_seed(42 + rank)  # the launcher's per-rank seed sequence
        ckpt_dir = ctx["checkpoint_dir"]
        pp_size = ctx.get("pp_size")
        pp_n_microbatches = ctx.get("pp_n_microbatches")
        amp_bf16 = ctx.get("amp_bf16", False)
        task_cls = ctx.get("task_cls") or LanguageModelingTask
        data_cls = _SFTDictDataModule if ctx.get("dict_batch") else DummyLMDataModule

        # ---- Epoch 1: train from scratch, save checkpoint ----
        config1 = _build_config(
            checkpoint_dir=ckpt_dir,
            epochs=1,
            resume_from_checkpoint=None,
            pp_size=pp_size,
            pp_n_microbatches=pp_n_microbatches,
            amp_bf16=amp_bf16,
        )
        data_module = data_cls(config1)
        data_module.prepare_data()
        data_module.setup()
        task = task_cls(config1, data_module)
        engine = TrainingEngine(config=config1, task=task, rank=rank, world_size=world_size, data_module=data_module)

        before = model_state_dict(engine.model)
        before_digests = _digests(before)
        engine.run()
        steps1 = engine.global_step
        full1 = model_state_dict(engine.model)
        digests1 = _digests(full1)
        _assert_ranks_bit_identical(rank, digests1, world_size, "epoch-1 training")
        assert steps1 > 0, f"rank {rank}: no training steps ran"
        moved = [key for key in before_digests if before_digests[key] != digests1.get(key)]
        assert moved, f"rank {rank}: no parameter changed across PP training — the run was a no-op"

        # ---- TASK-212: the checkpoint must carry THIS rank's optimizer ----#
        # ---- moments (Adam exp_avg), with an entry per stage/rank.      ---#
        ckpt_path = str(Path(ckpt_dir) / "epoch_1.pt")
        payload = load_checkpoint_payload(ckpt_path)
        optimizer_blob = payload.get("optimizer_state")
        assert isinstance(optimizer_blob, dict), "PP checkpoint must persist an optimizer_state"
        stage_states = optimizer_blob.get("__pp_stage_states__")
        assert isinstance(stage_states, dict), "PP checkpoint must persist per-stage optimizer states (TASK-212)"
        assert str(rank) in stage_states, f"rank {rank}: own stage optimizer state missing from checkpoint"
        saved_moment = _first_moment(stage_states[str(rank)])
        assert saved_moment is not None, "PP optimizer moments must be non-empty after training (TASK-212)"

        # ---- Epoch 2: resume from the checkpoint, continue ----
        # Every rank must load its OWN slice from the global-named dict; a
        # second engine run finishing in lockstep proves the collective save
        # and the scatter load agree. On rank 0 the saved checkpoint exists;
        # all ranks read the same files (shared filesystem).
        config2 = _build_config(
            checkpoint_dir=ckpt_dir,
            epochs=2,
            resume_from_checkpoint=ckpt_path,
            pp_size=pp_size,
            pp_n_microbatches=pp_n_microbatches,
            amp_bf16=amp_bf16,
        )
        data_module2 = data_cls(config2)
        task2 = task_cls(config2, data_module2)
        engine2 = TrainingEngine(config=config2, task=task2, rank=rank, world_size=world_size, data_module=data_module2)
        # The resumed engine must sense it starts at epoch index 1.
        assert engine2.start_epoch == 1, f"rank {rank}: resume did not sense start_epoch=1"
        # TASK-212: the resumed optimizer must have restored ITS OWN stage's
        # moments (the pre-fix behavior reset them to a fresh state, so the
        # first exp_avg would be None here).
        resumed_moment = _first_moment(engine2.optimizer.state_dict())
        assert resumed_moment is not None, f"rank {rank}: resume reset the optimizer moments (TASK-212)"
        assert torch.equal(resumed_moment, saved_moment), (
            f"rank {rank}: resumed moments differ from checkpoint (TASK-212)"
        )
        engine2.run()
        assert engine2.global_step > steps1, f"rank {rank}: resume did not advance global_step"
        full2 = model_state_dict(engine2.model)
        digests2 = _digests(full2)
        _assert_ranks_bit_identical(rank, digests2, world_size, "epoch-2 resume-continue")
        # The resumed run trained FURTHER on top of the resumed weights.
        assert digests2 != digests1, f"rank {rank}: resumed training did not move weights"

        results[rank] = {"success": True, "steps1": steps1, "steps2": engine2.global_step, "moved": len(moved)}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 — report worker failure in the parent
        import traceback

        results[rank] = {"success": False, "error": traceback.format_exc()}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_pp_engine_e2e(
    tmp_path,
    world_size: int,
    *,
    pp_size: int | None = None,
    pp_n_microbatches: int | None = None,
    amp_bf16: bool = False,
    task_cls: type | None = None,
    dict_batch: bool = False,
) -> None:
    port = _free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    ckpt_dir = str(tmp_path / "pp_checkpoints")

    manager = mp.Manager()
    results: dict = manager.dict()
    # Force CPU in the workers: start_method="spawn" gives each worker a fresh
    # interpreter, and clearing CUDA_VISIBLE_DEVICES for the duration of the
    # spawn call means the child's INITIAL env hides all GPUs before its module
    # imports can probe torch.cuda. A FORKED child would instead inherit this
    # pytest parent's already-initialised CUDA context, put the engine on a
    # CUDA device, and crash gloo (which transports CPU tensors only).
    saved_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        context = mp.spawn(
            _worker,
            args=(
                world_size,
                {
                    "checkpoint_dir": ckpt_dir,
                    "pp_size": pp_size,
                    "pp_n_microbatches": pp_n_microbatches,
                    "amp_bf16": amp_bf16,
                    "task_cls": task_cls,
                    "dict_batch": dict_batch,
                },
                results,
            ),
            nprocs=world_size,
            join=False,
            start_method="spawn",
        )
    finally:
        if saved_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_visible

    end_at = time.monotonic() + PP_ENGINE_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"PP engine e2e exceeded {PP_ENGINE_JOIN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break
    results = dict(results)
    assert len(results) == world_size, f"expected {world_size} ranks, got {len(results)}"
    for rank, payload in results.items():
        assert payload["success"], f"rank {rank} failed: {payload.get('error')}"
    step1 = {results[rank]["steps1"] for rank in results}
    step2 = {results[rank]["steps2"] for rank in results}
    assert len(step1) == 1, f"ranks must step in lockstep (epoch 1): {step1}"
    assert len(step2) == 1, f"ranks must step in lockstep (epoch 2): {step2}"
    assert results[0]["moved"] > 0
    assert results[0]["steps2"] > results[0]["steps1"]


def test_pp_engine_trains_and_resumes_two_process_cpu(tmp_path) -> None:
    _run_pp_engine_e2e(tmp_path, 2)


def test_pp_dp_2d_engine_trains_and_resumes_four_process_cpu(tmp_path) -> None:
    """PP + data-parallel 2D (RIL TASK-211) engine e2e on a 2x2 [DP][PP] grid.

    Four CPU/gloo ranks with ``pp_size=2``: the DP dimension shards the data
    per DP group, ``allreduce_pp_dp_grads`` averages at each step boundary,
    and the checkpoint path (collective save + scatter resume) must keep all
    four ranks' full state dicts bit-identical — the DP-group-divergence
    failure mode this milestone exists to prevent fails the equality check.
    """
    _run_pp_engine_e2e(tmp_path, 4, pp_size=2)


def test_pp_microbatch_overlap_engine_trains_and_resumes_two_process_cpu(tmp_path) -> None:
    """PP with n_microbatches=2 (RIL TASK-213) through the REAL engine.

    The schedule chunks each batch into two microbatches (different loss
    entries, gradient normalised internally) — the engine must still train,
    save and resume with all ranks' full state dicts bit-identical and the
    per-stage optimizer moments restored.
    """
    _run_pp_engine_e2e(tmp_path, 2, pp_n_microbatches=2)


def test_pp_bf16_amp_engine_trains_and_resumes_two_process_cpu(tmp_path) -> None:
    """PP + bf16 AMP (RIL TASK-214) through the REAL engine on CPU.

    ``use_amp=True`` with ``amp_dtype='bfloat16'`` runs every stage's forward
    AND the schedule's internal backward inside bf16 autocast — no GradScaler
    (bf16 needs no loss scaling, and float16 is refused because the schedule
    cannot scale losses before its internal backward). Train/save/resume must
    keep all ranks' bf16-trained weights bit-identical and restore the (fp32)
    optimizer moments.
    """
    _run_pp_engine_e2e(tmp_path, 2, amp_bf16=True)


def test_pp_engine_sft_alias_trains_and_resumes_two_process_cpu(tmp_path) -> None:
    """SFT alias under pipeline parallelism (RIL TASK-304 / ISS-339 overlap).

    ``SFTTask`` is a pure alias of ``LanguageModelingTask``, so it MUST run
    under ``parallel_strategy='pp'`` exactly like the parent. It also collates
    SFT-style dict batches carrying the extra ``attention_mask`` key, which the
    PP stage forward must drop without changing the loss (previously SFT's own
    train_step threaded that mask into the model, a mask flash_attn ignored and
    the PP scheduler dropped — behaviour differed by backend/strategy). Train /
    save / resume must keep both ranks' dicts bit-identical and restore the
    per-stage optimizer moments, same as the LM case.
    """
    _run_pp_engine_e2e(tmp_path, 2, task_cls=SFTTask, dict_batch=True)
