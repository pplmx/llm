"""Native ZeRO Stage-1 (``ZeroOptimizer``) tests (RIL TASK-267 / ISS-303).

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

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from llm.training.distributed import ZeroOptimizer

SEED = 123
IN_DIM = 4
HID_DIM = 5
OUT_DIM = 3
STEPS = 12
JOIN_TIMEOUT_S = 120
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
            "gloo", init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"], rank=rank, world_size=world_size
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
            "gloo", init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"], rank=rank, world_size=world_size
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
