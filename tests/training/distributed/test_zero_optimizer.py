"""Native ZeRO Stage-1 (``ZeroOptimizer``) tests.

Verification strategy: numeric parity on CPU + gloo (no GPU), mirroring the
TP/PP distributed tests. Two ranks build an identical ``nn.Module``; each wraps
a per-owner AdamW in :class:`ZeroOptimizer` (round-robin ownership), trains a
few steps on the same data, and after the parameter all-gather every rank's
weights must be BIT-EXACT to a single-process reference AdamW over the full
parameter set.

ZeRO Stage-1 assumption (shared with upstream DeepSpeed): gradients are
globally identical on every rank before ``step`` — trivially true here because
both ranks run the same data, and true under DDP after the gradient average.
"""

from __future__ import annotations

import os
import socket
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from llm.models.decoder import DecoderModel
from llm.training.distributed import ZeroOptimizer
from llm.training.distributed.zero import ZERO_RANK_STATES_KEY, ZeroStage2Optimizer
from llm.training.tasks.base_task import TrainingTask

# torch.multiprocessing.spawn multi-rank training: heavy (see
# test_pipeline_parallel.py); excluded from the per-push CPU gate.
pytestmark = pytest.mark.heavy

SEED = 123
IN_DIM = 4
HID_DIM = 5
OUT_DIM = 3
STEPS = 12
JOIN_TIMEOUT_S = 120
# Bounded per-collective timeout for the spawn workers' hand-rolled process
# groups (they call ``init_process_group`` directly rather than going through
# ``DistributedManager``, so they do NOT inherit
# ``DistributedConfig.collective_timeout_seconds``).
#
# Without it torch's default is 30 MINUTES, so a rank that dies mid-test leaves
# its siblings blocking far past the parent's join window and the parent reports
# a blind "ZeROOptimizer spawn exceeded" with no cause. ``timeout=`` bounds the initial
# RENDEZVOUS too, hence above a cold ``mp.spawn`` and under the parent join.
PG_COLLECTIVE_TIMEOUT_S = timedelta(seconds=100)
STEPS_PRE = 5
STEPS_POST = 7


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _setup_env() -> int:
    port = _free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    return port


def _build_model(seed: int = SEED) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(IN_DIM, HID_DIM), nn.Tanh(), nn.Linear(HID_DIM, OUT_DIM))


def _make_data() -> torch.Tensor:
    torch.manual_seed(999)
    return torch.randn(6, IN_DIM)


def _reference_params() -> list[torch.Tensor]:
    """Train the full parameter set with a plain AdamW for STEPS steps."""
    model = _build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = _make_data()
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
    return [p.detach().clone() for p in model.parameters()]


def _run_spawn(worker, world_size: int) -> dict:
    """Spawn ``worker(rank, world_size, results)`` over gloo/CPU procs."""
    _setup_env()
    manager = mp.Manager()
    results: dict = manager.dict()
    context = mp.spawn(worker, args=(world_size, results), nprocs=world_size, join=False)
    end_at = time.monotonic() + JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"ZeroOptimizer spawn exceeded {JOIN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break
    out = dict(results)
    if any(v.get("error") for v in out.values()):
        first = next(v["error"] for v in out.values() if v.get("error"))
        raise AssertionError(f"ZeroOptimizer worker failed: {first}")
    return out


def _parity_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"],
            rank=rank,
            world_size=world_size,
            timeout=PG_COLLECTIVE_TIMEOUT_S,
        )
        model = _build_model()
        params = list(model.parameters())
        optimizer = ZeroOptimizer(
            lambda owned: torch.optim.AdamW(owned, lr=0.01),
            params,
            rank=rank,
            world_size=world_size,
        )
        owned_here = len([i for i in range(len(params)) if i % world_size == rank])
        assert optimizer.owned_count() == owned_here, "rank must own exactly its round-robin shard"

        x = _make_data()
        for _ in range(STEPS):
            optimizer.zero_grad()
            loss = model(x).square().mean()
            loss.backward()
            optimizer.step()

        # After the final all-gather, this rank holds the fully updated params.
        results[rank] = {"params": [p.detach().clone() for p in params], "error": None}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


@pytest.mark.quick
def test_zero_stage1_parity_two_process_cpu() -> None:
    """Sharded per-owner AdamW + all-gather matches a full AdamW bit-exactly."""
    reference = _reference_params()
    out = _run_spawn(_parity_worker, 2)
    for rank, payload in out.items():
        assert payload.get("error") is None, f"rank {rank}: {payload.get('error')}"
        assert len(payload["params"]) == len(reference)
        for i, (got, want) in enumerate(zip(payload["params"], reference, strict=True)):
            assert torch.equal(got, want), f"rank {rank} param {i} not bit-exact to reference"


@pytest.mark.quick
def test_zero_optimizer_single_rank_matches_plain_optimizer() -> None:
    """world_size=1 ZeroOptimizer is behaviourally identical to a plain AdamW."""
    model = _build_model()
    optimizer = ZeroOptimizer(
        lambda owned: torch.optim.AdamW(owned, lr=0.01),
        model.parameters(),
        rank=0,
        world_size=1,
    )
    x = _make_data()
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
    got = [p.detach().clone() for p in model.parameters()]
    for p, want in zip(got, _reference_params(), strict=True):
        assert torch.equal(p, want)


def _reference_checkpoint_params() -> list[torch.Tensor]:
    """Train STEPS_PRE with a full AdamW, dump its state, reset to init, reload,
    then continue STEPS_POST — the reference for a sharded resume."""
    torch.manual_seed(SEED)
    model = _build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = _make_data()
    for _ in range(STEPS_PRE):
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
    saved = optimizer.state_dict()

    torch.manual_seed(SEED)
    reset = _build_model()
    resumed = torch.optim.AdamW(reset.parameters(), lr=0.01)
    resumed.load_state_dict(saved)
    for _ in range(STEPS_POST):
        resumed.zero_grad()
        reset(x).square().mean().backward()
        resumed.step()
    return [p.detach().clone() for p in reset.parameters()]


def _checkpoint_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"],
            rank=rank,
            world_size=world_size,
            timeout=PG_COLLECTIVE_TIMEOUT_S,
        )
        x = _make_data()

        # Phase 1: train STEPS_PRE with a sharded optimizer, dump its state.
        model = _build_model()
        optimizer = ZeroOptimizer(
            lambda owned: torch.optim.AdamW(owned, lr=0.01),
            model.parameters(),
            rank=rank,
            world_size=world_size,
        )
        for _ in range(STEPS_PRE):
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
        saved = optimizer.state_dict()
        assert saved["_zero_stage1"]
        assert saved["rank"] == rank
        assert saved["inner"]["state"], "sharded AdamW must have captured nonzero moments"

        # Phase 2: reset to init, reload this rank's shard, continue STEPS_POST.
        torch.manual_seed(SEED)
        reset = _build_model()
        resumed = ZeroOptimizer(
            lambda owned: torch.optim.AdamW(owned, lr=0.01),
            reset.parameters(),
            rank=rank,
            world_size=world_size,
        )
        resumed.load_state_dict(saved)
        for _ in range(STEPS_POST):
            resumed.zero_grad()
            reset(x).square().mean().backward()
            resumed.step()
        results[rank] = {"params": [p.detach().clone() for p in reset.parameters()], "error": None}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


@pytest.mark.quick
def test_zero_stage1_checkpoint_roundtrip_two_process_cpu() -> None:
    """Per-rank sharded optimizer state survives save/reset/load cumulatively."""
    reference = _reference_checkpoint_params()
    out = _run_spawn(_checkpoint_worker, 2)
    for rank, payload in out.items():
        assert payload.get("error") is None, f"rank {rank}: {payload.get('error')}"
        for i, (got, want) in enumerate(zip(payload["params"], reference, strict=True)):
            assert torch.equal(got, want), f"rank {rank} param {i} diverged after checkpoint resume"


def _engine_zero_worker(rank: int, world_size: int, results: dict) -> None:
    """Drive the real TrainingEngine with parallel_strategy='zero' (2-rank gloo)."""
    try:
        # Force the engine down the CPU path (mirrors the PP engine e2e): a
        # forked child can inherit a live CUDA context, and gloo only transports
        # CPU tensors.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"],
            rank=rank,
            world_size=world_size,
            timeout=PG_COLLECTIVE_TIMEOUT_S,
        )
        from llm.training.core.config import (
            Config,
            DistributedConfig,
            ModelConfig,
            OptimizationConfig,
            TrainingConfig,
        )
        from llm.training.core.engine import TrainingEngine
        from llm.training.distributed.zero import ZeroOptimizer
        from llm.training.tasks.lm_task import LanguageModelingTask
        from tests.support.data import DummyLMDataModule

        config = Config(
            model=ModelConfig(vocab_size=64, hidden_size=16, num_layers=2, num_heads=2, max_seq_len=32),
            training=TrainingConfig(batch_size=4, epochs=2, num_samples=32),
            optimization=OptimizationConfig(use_amp=False, use_compile=False),
            distributed=DistributedConfig(backend="gloo", parallel_strategy="zero"),
        )
        data_module = DummyLMDataModule(config)
        task = LanguageModelingTask(config, data_module)
        engine = TrainingEngine(config, task, rank=rank, world_size=world_size, data_module=data_module)

        assert engine.is_training_zero
        assert isinstance(engine.optimizer, ZeroOptimizer)
        # round-robin ownership must be non-empty but strictly smaller than the
        # full set on each rank (a genuine shard); the parent asserts the two
        # ranks' shards are disjoint and cover every param.
        total = len(list(engine.model.parameters()))
        owned = engine.optimizer.owned_count()
        assert 0 < owned < total

        loss = engine._run_epoch(0)
        results[rank] = {
            "loss": float(loss),
            "params": [p.detach().clone() for p in engine.model.parameters()],
            "owned": owned,
            "total": total,
            "error": None,
        }
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


@pytest.mark.quick
def test_zero_stage1_engine_standard_loop_two_process_cpu() -> None:
    """TrainingEngine parallel_strategy='zero' trains with finite loss and keeps
    both ranks bit-exact after the gradient-average + sharded-step + all-gather."""
    out = _run_spawn(_engine_zero_worker, 2)
    losses: list[float] = []
    for rank, payload in out.items():
        assert payload.get("error") is None, f"rank {rank}: {payload.get('error')}"
        assert payload["loss"] < float("inf"), f"rank {rank}: loss NaN/inf"
        losses.append(payload["loss"])
    assert out[0]["total"] == out[1]["total"]
    # disjoint round-robin shards cover exactly the full parameter set.
    assert out[0]["owned"] + out[1]["owned"] == out[0]["total"]
    # Both ranks trained the same full-batch gradient and all-gathered the
    # updated weights, so their parameters must be bit-identical.
    for p0, p1 in zip(out[0]["params"], out[1]["params"], strict=True):
        assert torch.equal(p0, p1), "zero ranks drifted out of lockstep"


# ---------------------------------------------------------------------------
# Custom-loop refusal for the multi-rank parallel strategies
# ---------------------------------------------------------------------------


class _CustomLoopTaskLike(TrainingTask):
    """A PPO-shaped task: full standard-loop API surface, custom driver.

    ``PPOTask`` is the production example (``uses_standard_loop = False``), but
    building one needs a prompt corpus, a tokenizer and a reward model. The
    engine's guard reads only ``task.uses_standard_training_loop()`` and fires
    during ``_setup_components`` before any task method runs, so marking a real
    task custom is enough — and it keeps the test about the guard rather than
    about PPO's setup.
    """

    uses_standard_loop = False

    def build_model(self):
        return DecoderModel(vocab_size=16, hidden_size=16, num_layers=2, num_heads=2, max_seq_len=16)

    def build_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def build_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.1)

    def build_scheduler(self, optimizer):
        return None

    def train_step(self, batch, model, criterion):
        return torch.tensor(0.0), {"loss": 0.0}

    def validation_step(self, batch, model, criterion):
        return torch.tensor(0.0), {"loss": 0.0}

    def run_training(self, engine):
        return None


def _refusal_worker(rank: int, world_size: int, results: dict) -> None:
    """Record the engine's exception message for the strategy in ``GUARD_STRATEGY``."""
    try:
        strategy = os.environ["GUARD_STRATEGY"]
        dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=PG_COLLECTIVE_TIMEOUT_S)

        from llm.data.modules.synthetic import SyntheticDataModule
        from llm.training.core.config import (
            Config,
            DistributedConfig,
            LoggingConfig,
            ModelConfig,
            OptimizationConfig,
            TrainingConfig,
        )
        from llm.training.core.engine import TrainingEngine

        config = Config(
            model=ModelConfig(vocab_size=16, hidden_size=16, num_layers=2, num_heads=2, max_seq_len=16),
            training=TrainingConfig(batch_size=2, num_samples=8, epochs=1, run_validation=False),
            optimization=OptimizationConfig(num_workers=0, use_compile=False, use_amp=False),
            distributed=DistributedConfig(
                parallel_strategy=strategy,
                gpus_per_node=world_size,
                pp_size=world_size if strategy in ("pp", "3d") else 0,
                tp_size=1 if strategy == "3d" else 0,
                dp_size=1,
            ),
            logging=LoggingConfig(log_interval=10, log_level="ERROR", save_logs=False),
        )
        data_module = SyntheticDataModule(config)
        data_module.setup()
        TrainingEngine(
            config, _CustomLoopTaskLike(config, data_module), rank=rank, world_size=world_size, data_module=data_module
        )
        results[rank] = {"error": None, "message": "NOT REFUSED"}
    except Exception as exc:  # noqa: BLE001 — the message is the assertion payload
        results[rank] = {"error": type(exc).__name__, "message": str(exc)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_refusal(world_size: int, strategy: str) -> dict:
    """Spawn the refusal probe and return the per-rank payloads.

    Unlike :func:`_run_spawn` this must NOT raise on a worker error: the
    recorded exception message IS the result under test.
    """
    os.environ["GUARD_STRATEGY"] = strategy
    _setup_env()
    manager = mp.Manager()
    results: dict = manager.dict()
    context = mp.spawn(_refusal_worker, args=(world_size, results), nprocs=world_size, join=False)
    end_at = time.monotonic() + JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"custom-loop refusal probe exceeded {JOIN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break
    out = dict(results)
    manager.shutdown()
    return out


@pytest.mark.quick
def test_zero_refuses_a_custom_loop_task() -> None:
    """ZeRO Stage-1 v1 drives only the standard loop.

    ``docs/guides/distributed.md`` states this as the v1 status ("only supports
    standard-loop tasks"), and the engine enforces it with a named message — but
    nothing tested the promise, so a refactor could have silently removed the
    check and left a custom-loop task to run under a sharded optimizer whose
    step the engine never drives.
    """
    out = _run_refusal(2, "zero")
    for rank, payload in out.items():
        assert payload["error"] is not None, f"rank {rank}: ZeRO accepted a custom-loop task"
        assert "only supports standard-loop tasks" in payload["message"], payload["message"]
        assert "custom training loop" in payload["message"], payload["message"]


@pytest.mark.quick
def test_pp_refuses_a_custom_loop_task() -> None:
    """PP's loss contract is the LM-shift + CE one, so a custom loop cannot fit.

    This guard sits AFTER ``wrap_model_for_training``, which is why the test
    needs a live process group: the wrap is where a PP model is partitioned, and
    it raises the generic "Default process group has not been initialized"
    error before the guard could speak. With the group up, the guard runs and
    names the real reason.
    """
    out = _run_refusal(2, "pp")
    for rank, payload in out.items():
        assert payload["error"] is not None, f"rank {rank}: PP accepted a custom-loop task"
        assert "only supports standard-loop tasks" in payload["message"], payload["message"]


# --- ZeRO Stage-2 (ADR-014) ------------------------------------------------
#
# Stage-2 shards gradients as well as optimizer state. The property that matters
# is not "it runs" but "it produces the SAME weights as DDP + Stage-1", since
# the shard math must not change the optimization. These tests hold it to the
# bit-exact standard the Stage-1 tests already use.


def _stage2_worker(rank: int, world_size: int, results: dict) -> None:
    """Train with Stage-2: reduce grads to owners, clip globally, then step."""
    try:
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"],
            rank=rank,
            world_size=world_size,
            timeout=PG_COLLECTIVE_TIMEOUT_S,
        )
        model = _build_model()
        params = list(model.parameters())
        optimizer = ZeroStage2Optimizer(
            lambda owned: torch.optim.AdamW(owned, lr=0.01),
            params,
            rank=rank,
            world_size=world_size,
        )
        x = _make_data()
        for _ in range(STEPS):
            optimizer.zero_grad()
            model(x).square().mean().backward()
            # ADR-014: clip on the GLOBAL norm BEFORE the reduction.
            optimizer.clip_gradients_globally(max_norm=0.0)  # 0 = no clipping
            optimizer.reduce_gradients_to_owners()
            optimizer.step()

        # Non-owned gradients must be gone after the reduction.
        non_owned_with_grad = [
            i for i, owner in enumerate(optimizer._owners) if owner != rank and params[i].grad is not None
        ]
        results[rank] = {
            "params": [p.detach().clone() for p in params],
            "non_owned_with_grad": non_owned_with_grad,
            "error": None,
        }
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


@pytest.mark.quick
def test_zero_stage2_matches_full_ddp_reference_bit_exactly() -> None:
    """Stage-2 (gradient sharding) must reach the same weights as a full AdamW.

    The reference is the same one the Stage-1 parity test uses: a single-process
    full AdamW over the whole gradient. Stage-2 reaches it through a different
    path — each rank keeps only its owned gradients after a reduce_scatter — so
    agreement proves the reduction is a true cross-rank MEAN and that dropping
    the non-owned buffers changes nothing.
    """
    reference = _reference_params()
    out = _run_spawn(_stage2_worker, 2)
    for rank, payload in out.items():
        assert payload.get("error") is None, f"rank {rank}: {payload.get('error')}"
        assert payload["non_owned_with_grad"] == [], (
            f"rank {rank} kept gradients for non-owned params {payload['non_owned_with_grad']}; "
            "Stage-2 must drop them or the memory saving is lost"
        )
        for i, (got, want) in enumerate(zip(payload["params"], reference, strict=True)):
            assert torch.equal(got, want), (
                f"rank {rank} param {i} not bit-exact to the full-AdamW reference "
                f"(max |d|={(got - want).abs().max().item():.3e})"
            )


def _stage2_reduction_worker(rank: int, world_size: int, results: dict) -> None:
    """Check the reduced gradient equals the cross-rank mean, directly."""
    try:
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"],
            rank=rank,
            world_size=world_size,
            timeout=PG_COLLECTIVE_TIMEOUT_S,
        )
        model = _build_model()
        params = list(model.parameters())
        optimizer = ZeroStage2Optimizer(
            lambda owned: torch.optim.SGD(owned, lr=0.0),  # lr 0: step is a no-op
            params,
            rank=rank,
            world_size=world_size,
        )
        # Give each rank a DISTINCT gradient so the mean is not trivially equal
        # to either contribution.
        for i, p in enumerate(params):
            p.grad = torch.full_like(p, float(rank + 1) * float(i + 1))

        expected = {}
        for i in range(len(params)):
            expected[i] = sum(float(r + 1) * float(i + 1) for r in range(world_size)) / world_size

        optimizer.reduce_gradients_to_owners()

        got = {}
        for i, owner in enumerate(optimizer._owners):
            if owner == rank:
                grad = params[i].grad
                got[i] = None if grad is None else float(grad.flatten()[0].item())
            else:
                got[i] = "dropped" if params[i].grad is None else "KEPT"
        results[rank] = {"got": got, "expected": expected, "error": None}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        import traceback

        results[rank] = {"error": traceback.format_exc()}


@pytest.mark.quick
def test_zero_stage2_reduction_is_a_cross_rank_mean() -> None:
    """Each owner sees the mean of the per-rank gradients, and nothing else."""
    out = _run_spawn(_stage2_reduction_worker, 2)
    for rank, payload in out.items():
        assert payload.get("error") is None, f"rank {rank}: {payload.get('error')}"
        got, expected = payload["got"], payload["expected"]
        for index, value in got.items():
            if value == "dropped":
                continue
            assert value != "KEPT", f"rank {rank} kept a non-owned gradient for param {index}"
            assert value == pytest.approx(expected[index], abs=1e-5), (
                f"rank {rank} param {index}: reduced grad {value} != cross-rank mean {expected[index]}"
            )


def _stage2_clip_worker(rank: int, world_size: int, results: dict) -> None:
    """Global clipping must use ONE factor across ranks (ADR-014)."""
    try:
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"],
            rank=rank,
            world_size=world_size,
            timeout=PG_COLLECTIVE_TIMEOUT_S,
        )
        model = _build_model()
        params = list(model.parameters())
        optimizer = ZeroStage2Optimizer(
            lambda owned: torch.optim.SGD(owned, lr=0.0),
            params,
            rank=rank,
            world_size=world_size,
        )
        # Very different scales per rank: local norms would differ wildly, so a
        # per-rank clip would not be the same factor.
        scale = 10.0 if rank == 0 else 0.1
        for p in params:
            p.grad = torch.ones_like(p) * scale
        local_norm = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in params)))

        global_norm = optimizer.clip_gradients_globally(max_norm=1.0)

        after = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in params)))
        results[rank] = {
            "local_norm": local_norm,
            "global_norm": float(global_norm),
            "after_norm": after,
            "error": None,
        }
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        import traceback

        results[rank] = {"error": traceback.format_exc()}


@pytest.mark.quick
def test_zero_stage2_global_clip_uses_one_factor() -> None:
    """The reported norm is the GLOBAL one on every rank, and both scale by it."""
    out = _run_spawn(_stage2_clip_worker, 2)
    for rank, payload in out.items():
        assert payload.get("error") is None, f"rank {rank}: {payload.get('error')}"

    global_norms = [out[r]["global_norm"] for r in (0, 1)]
    assert global_norms[0] == pytest.approx(global_norms[1], rel=1e-6), (
        f"ranks disagree on the global norm ({global_norms}); each must all-reduce it"
    )
    # Each rank's post-clip norm must be its ORIGINAL norm scaled by the SAME
    # factor (1/global_norm), not independently clipped to 1.0.
    factors = [out[r]["after_norm"] / out[r]["local_norm"] for r in (0, 1)]
    assert factors[0] == pytest.approx(factors[1], rel=1e-5), (
        f"the clip factors differ per rank ({factors}); that is the desync ADR-014 warns about"
    )
    expected = 1.0 / global_norms[0]
    assert factors[0] == pytest.approx(expected, rel=1e-5)


# --- Stage-2 through the real engine (ADR-014) ----------------------------


def _engine_zero_stage2_worker(rank: int, world_size: int, results: dict) -> None:
    """Drive TrainingEngine with parallel_strategy='zero', zero_stage=2."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU path; gloo only transports CPU tensors
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"],
            rank=rank,
            world_size=world_size,
            timeout=PG_COLLECTIVE_TIMEOUT_S,
        )
        from llm.training.core.config import (
            Config,
            DistributedConfig,
            ModelConfig,
            OptimizationConfig,
            TrainingConfig,
        )
        from llm.training.core.engine import TrainingEngine
        from llm.training.tasks.lm_task import LanguageModelingTask
        from tests.support.data import DummyLMDataModule

        config = Config(
            model=ModelConfig(vocab_size=64, hidden_size=16, num_layers=2, num_heads=2, max_seq_len=32),
            training=TrainingConfig(batch_size=4, epochs=2, num_samples=32),
            optimization=OptimizationConfig(use_amp=False, use_compile=False),
            distributed=DistributedConfig(backend="gloo", parallel_strategy="zero", zero_stage=2),
        )
        data_module = DummyLMDataModule(config)
        task = LanguageModelingTask(config, data_module)
        engine = TrainingEngine(config, task, rank=rank, world_size=world_size, data_module=data_module)

        assert engine.is_training_zero
        assert engine.zero_stage == 2
        assert isinstance(engine.optimizer, ZeroStage2Optimizer), (
            f"zero_stage=2 built {type(engine.optimizer).__name__}, not a ZeroStage2Optimizer"
        )

        loss = engine._run_epoch(0)
        results[rank] = {
            "loss": float(loss),
            "params": [p.detach().clone() for p in engine.model.parameters()],
            "error": None,
        }
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


@pytest.mark.quick
def test_zero_stage2_engine_standard_loop_two_process_cpu() -> None:
    """The real engine selects Stage-2 from config, trains, and keeps ranks in lockstep.

    Stage-1's engine test already covers "sharded optimizer inside the training
    loop". What is new here is the DISPATCH: `zero_stage=2` must build a
    `ZeroStage2Optimizer` and take the clip-then-reduce branch, and the two ranks
    must end bit-exact (the parameter all-gather still restores identical
    weights, exactly as in Stage-1).
    """
    out = _run_spawn(_engine_zero_stage2_worker, 2)
    for rank, payload in out.items():
        assert payload.get("error") is None, f"rank {rank}: {payload.get('error')}"
        assert payload["loss"] < float("inf"), f"rank {rank}: loss NaN/inf"

    ref = out[0]["params"]
    for rank in (1,):
        for i, (got, want) in enumerate(zip(out[rank]["params"], ref, strict=True)):
            assert torch.equal(got, want), (
                f"rank {rank} param {i} diverged from rank 0 after a Stage-2 epoch "
                f"(max |d|={(got - want).abs().max().item():.3e})"
            )


def _stage2_grads_are_sharded_worker(rank: int, world_size: int, results: dict) -> None:
    """After one Stage-2 step, only OWNED params may still hold a gradient."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"],
            rank=rank,
            world_size=world_size,
            timeout=PG_COLLECTIVE_TIMEOUT_S,
        )
        import torch.nn as nn

        from llm.training.distributed.zero import ZeroStage2Optimizer

        torch.manual_seed(SEED)
        model = nn.Sequential(nn.Linear(IN_DIM, HID_DIM), nn.Tanh(), nn.Linear(HID_DIM, OUT_DIM))
        params = list(model.parameters())
        optimizer = ZeroStage2Optimizer(
            lambda owned: torch.optim.SGD(owned, lr=0.01),
            params,
            rank=rank,
            world_size=world_size,
        )
        optimizer.zero_grad()
        model(_make_data()).square().mean().backward()
        optimizer.clip_gradients_globally(0.0)
        optimizer.reduce_gradients_to_owners()

        # Nothing has stepped yet, so the sharding is observable directly.
        with_grad = [i for i, p in enumerate(params) if p.grad is not None]
        owned = [i for i, owner in enumerate(optimizer._owners) if owner == rank]
        results[rank] = {"with_grad": with_grad, "owned": owned, "error": None}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        import traceback

        results[rank] = {"error": traceback.format_exc()}


@pytest.mark.quick
def test_zero_stage2_leaves_only_owned_gradients() -> None:
    """The memory claim, observed through the real optimizer: non-owned grads gone."""
    out = _run_spawn(_stage2_grads_are_sharded_worker, 2)
    seen_owned: set[int] = set()
    for rank, payload in out.items():
        assert payload.get("error") is None, f"rank {rank}: {payload.get('error')}"
        assert payload["with_grad"] == payload["owned"], (
            f"rank {rank} holds gradients {payload['with_grad']} but owns {payload['owned']}; "
            "Stage-2 must drop every non-owned gradient or the memory saving is lost"
        )
        seen_owned |= set(payload["owned"])
    assert len(seen_owned) == 4, f"the two ranks' shards must cover all 4 params, got {sorted(seen_owned)}"


# --- Stage-2 checkpoint roundtrip (ADR-014) -------------------------------


def _reference_checkpoint_params_stage2() -> list[torch.Tensor]:
    """Reference for the Stage-2 resume: the same math a full optimizer does.

    Stage-2 changes WHERE gradients live, not what the update is, so the
    reference is identical to the Stage-1 one: train STEPS_PRE full, dump, reset,
    reload, continue STEPS_POST.
    """
    return _reference_checkpoint_params()


def _stage2_checkpoint_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"],
            rank=rank,
            world_size=world_size,
            timeout=PG_COLLECTIVE_TIMEOUT_S,
        )
        x = _make_data()

        # Phase 1: train STEPS_PRE with Stage-2, save THE GATHERED payload.
        model = _build_model()
        optimizer = ZeroStage2Optimizer(
            lambda owned: torch.optim.AdamW(owned, lr=0.01),
            model.parameters(),
            rank=rank,
            world_size=world_size,
        )
        for _ in range(STEPS_PRE):
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.clip_gradients_globally(0.0)
            optimizer.reduce_gradients_to_owners()
            optimizer.step()

        per_rank_shard = optimizer.state_dict()
        assert per_rank_shard["zero_stage"] == 2, "the shard must record the stage that wrote it"
        # The multi-rank resume path persists the GATHERED envelope; it is a
        # collective, so every rank must enter it.
        payload = optimizer.collective_state_dict()
        assert ZERO_RANK_STATES_KEY in payload
        assert len(payload[ZERO_RANK_STATES_KEY]) == world_size, (
            "the gathered envelope must hold one shard per rank, or a multi-rank resume "
            "would silently cold-start the missing shards"
        )

        # Phase 2: reset to init, reload this rank's slice by its OWN rank key.
        torch.manual_seed(SEED)
        reset = _build_model()
        resumed = ZeroStage2Optimizer(
            lambda owned: torch.optim.AdamW(owned, lr=0.01),
            reset.parameters(),
            rank=rank,
            world_size=world_size,
        )
        resumed.load_state_dict(payload)
        for _ in range(STEPS_POST):
            resumed.zero_grad()
            reset(x).square().mean().backward()
            resumed.clip_gradients_globally(0.0)
            resumed.reduce_gradients_to_owners()
            resumed.step()
        results[rank] = {"params": [p.detach().clone() for p in reset.parameters()], "error": None}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


@pytest.mark.quick
def test_zero_stage2_checkpoint_roundtrip_two_process_cpu() -> None:
    """Stage-2 sharded moments survive save/reset/load cumulatively.

    Stage-1 has this test; Stage-2 needs its own because the resume path also has
    to rebuild a ``ZeroStage2Optimizer`` (not a ``ZeroOptimizer``) and keep taking
    the clip-then-reduce branch. Agreement with the full-optimizer reference
    proves the restored moments are the real Adam state, not a cold restart.
    """
    reference = _reference_checkpoint_params_stage2()
    out = _run_spawn(_stage2_checkpoint_worker, 2)
    for rank in (0, 1):
        payload = out[rank]
        assert payload.get("error") is None, f"rank {rank}: {payload.get('error')}"
        for i, (got, want) in enumerate(zip(payload["params"], reference, strict=True)):
            assert torch.equal(got, want), (
                f"rank {rank} param {i} diverged after a Stage-2 checkpoint resume "
                f"(max |d|={(got - want).abs().max().item():.3e})"
            )


def test_zero_cross_stage_resume_is_refused() -> None:
    """A Stage-1 shard must not load into a Stage-2 job, or vice versa.

    The two stages write the SAME on-disk shape (owned-parameter moments), so
    nothing about the payload distinguishes them except the recorded stage. A
    resume across stages would silently change the gradient-reduction path —
    Stage-1 keeps and averages every gradient, Stage-2 drops the non-owned ones —
    so the training dynamics after resume would not match the run that was
    interrupted.
    """

    def build(cls, seed: int = 0):
        torch.manual_seed(seed)
        model = torch.nn.Linear(3, 2)
        opt = cls(lambda owned: torch.optim.AdamW(owned, lr=0.01), model.parameters(), rank=0, world_size=1)
        return model, opt

    stage1_model, stage1_opt = build(ZeroOptimizer)
    for p in stage1_model.parameters():
        p.grad = torch.ones_like(p)
    stage1_opt.step()
    stage1_shard = stage1_opt.state_dict()
    assert stage1_shard["zero_stage"] == 1

    _model2, stage2_opt = build(ZeroStage2Optimizer)
    assert stage2_opt.state_dict()["zero_stage"] == 2

    with pytest.raises(ValueError, match="written by ZeRO Stage-1 but the job is running Stage-2"):
        stage2_opt.load_state_dict(stage1_shard)

    # And the other direction.
    _model3, stage1_opt_b = build(ZeroOptimizer)
    with pytest.raises(ValueError, match="written by ZeRO Stage-2 but the job is running Stage-1"):
        stage1_opt_b.load_state_dict(stage2_opt.state_dict())


def test_a_pre_stage_marker_checkpoint_still_loads_as_stage1() -> None:
    """Checkpoints written before the stage key existed must stay resumable.

    They are Stage-1 by construction (Stage-2 did not exist), so an absent key
    defaults to 1 rather than failing every older checkpoint.
    """
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    opt = ZeroOptimizer(lambda owned: torch.optim.AdamW(owned, lr=0.01), model.parameters(), rank=0, world_size=1)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    legacy = opt.state_dict()
    del legacy["zero_stage"]  # what an older release wrote

    torch.manual_seed(0)
    fresh_model = torch.nn.Linear(3, 2)
    fresh = ZeroOptimizer(
        lambda owned: torch.optim.AdamW(owned, lr=0.01), fresh_model.parameters(), rank=0, world_size=1
    )
    fresh.load_state_dict(legacy)  # must not raise
    assert fresh.optimizer.state, "the restored shard should carry the moments"
