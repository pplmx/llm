"""The collective primitives ADR-014 relies on, verified independently of Stage-2.

ADR-014 (`docs/adr/014-zero-stage-2-activation-sharding.md`) decides to build
ZeRO Stage-2 natively on `reduce_scatter`. Before writing `ZeroStage2Optimizer`,
three assumptions in that design were measured; these tests keep them true so the
implementation is not built on a drifted primitive.

1. `reduce_scatter_single(out, full, op=SUM)` exists and means "sum the full
   buffers across ranks, then give each rank its own contiguous segment".
   Dividing by `world_size` yields the GLOBAL AVERAGE gradient that Stage-1's
   DDP-averaged contract assumes.
2. `reduce_scatter_tensor` is deprecated in this torch — using it emits a
   `FutureWarning`. The ADR originally named it, so the name is pinned here too.
3. Clipping must happen BEFORE the scatter, on the global norm. A per-rank local
   clip applies a DIFFERENT factor to each shard, which is the desync the ADR
   warns about.

They run on gloo/CPU with `mp.spawn`, matching the rest of
`tests/training/distributed/` — no GPU needed.
"""

from __future__ import annotations

import os
import time
import warnings
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.heavy

SPAWN_TIMEOUT_S = 120


def _run(worker, world_size: int = 2) -> dict:
    """Spawn ``worker`` over gloo and return its ``results`` dict."""
    manager = mp.Manager()
    results = manager.dict()
    # The store owns the port from allocation to close, so no other test can be
    # handed the same one (see tests/support/distributed.py).
    store = dist.TCPStore(
        "127.0.0.1", 0, world_size, is_master=True, wait_for_workers=False, timeout=timedelta(seconds=SPAWN_TIMEOUT_S)
    )
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(store.port)
    context = mp.spawn(worker, args=(world_size, results), nprocs=world_size, join=False)
    deadline = time.monotonic() + SPAWN_TIMEOUT_S
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"spawn exceeded {SPAWN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break
    del store
    out = dict(results)
    manager.shutdown()
    for rank, payload in out.items():
        if isinstance(payload, dict) and payload.get("error"):
            raise AssertionError(f"rank {rank}: {payload['error']}")
    return out


def _init(rank: int, world_size: int) -> None:
    # Join the parent's store (published via MASTER_PORT): the parent keeps it
    # open for the whole spawn, so the port is never free for another test to
    # take. A bare env-var rendezvous would look for a different server.
    store = dist.TCPStore(
        os.environ["MASTER_ADDR"],
        int(os.environ["MASTER_PORT"]),
        world_size,
        is_master=False,
        timeout=timedelta(seconds=SPAWN_TIMEOUT_S // 2),
    )
    dist.init_process_group(
        "gloo",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=SPAWN_TIMEOUT_S // 2),
    )


# --- 1. reduce_scatter_single semantics ------------------------------------


def _scatter_sum_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        _init(rank, world_size)
        per_rank = 4
        # Every rank holds a FULL buffer; rank r fills it with r+1.
        full = torch.full((world_size * per_rank,), float(rank + 1))
        owned = torch.empty(per_rank)
        dist.reduce_scatter_single(owned, full, op=dist.ReduceOp.SUM)
        results[rank] = {"owned": owned.tolist(), "sum_expected": float(sum(range(1, world_size + 1)))}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - surfaced by _run
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def test_reduce_scatter_single_sums_then_slices():
    """Each rank receives its segment of the cross-rank SUM."""
    out = _run(_scatter_sum_worker)
    expected_sum = out[0]["sum_expected"]
    for rank in (0, 1):
        assert out[rank]["owned"] == [expected_sum] * 4, (
            f"rank {rank} got {out[rank]['owned']}, expected every element = {expected_sum}. "
            "reduce_scatter_single must sum the full buffers and hand back the owned slice."
        )


def _average_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        _init(rank, world_size)
        per_rank = 3
        # rank 0 contributes 1.0, rank 1 contributes 3.0 -> mean is 2.0.
        local = torch.full((per_rank,), float(1 + 2 * rank))
        full = torch.cat([local.clone() for _ in range(world_size)])
        owned = torch.empty(per_rank)
        dist.reduce_scatter_single(owned, full, op=dist.ReduceOp.SUM)
        owned /= world_size
        results[rank] = {"mean": owned.tolist()}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def test_sum_then_divide_gives_the_global_average():
    """The recipe Stage-2 needs: SUM then / world_size == the DDP-averaged grad."""
    out = _run(_average_worker)
    for rank in (0, 1):
        assert out[rank]["mean"] == pytest.approx([2.0, 2.0, 2.0]), (
            f"rank {rank} averaged to {out[rank]['mean']}; 1.0 and 3.0 must average to 2.0"
        )


# --- 2. the deprecated name ------------------------------------------------


def _deprecation_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        _init(rank, world_size)
        per_rank = 2
        full = torch.ones(world_size * per_rank)
        owned = torch.empty(per_rank)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dist.reduce_scatter_tensor(owned, full, op=dist.ReduceOp.SUM)
        results[rank] = {"warnings": [str(w.message) for w in caught]}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def test_reduce_scatter_tensor_is_deprecated():
    """ADR-014 originally named this API; it warns, so the ADR names the new one.

    If a future torch un-deprecates or removes it, this test fails and the ADR's
    wording has to be revisited — which is the point of pinning a deprecation.
    """
    out = _run(_deprecation_worker)
    messages = " ".join(out[0]["warnings"])
    assert "reduce_scatter_tensor" in messages, (
        f"expected the warning to name the deprecated function, got: {out[0]['warnings']}"
    )
    assert "deprecat" in messages.lower(), (
        f"expected a deprecation warning for reduce_scatter_tensor, got: {out[0]['warnings']}"
    )


# --- 3. local clipping desynchronises shards ------------------------------


def test_local_clipping_applies_a_different_factor_per_rank():
    """Why ADR-014 requires clipping on the GLOBAL norm before the scatter.

    Two ranks whose local gradients differ in scale each compute their own norm.
    Clipping locally gives each a different multiplier, so the shards stop being
    updates of the same global step. A single global norm applies one factor.
    """
    torch.manual_seed(0)
    per_rank_grads = []
    for rank in range(2):
        torch.manual_seed(100 + rank)
        scale = 10.0 if rank == 0 else 0.1
        per_rank_grads.append([torch.randn(8, 8) * scale, torch.randn(2, 8) * scale])

    local_norms = [torch.sqrt(sum((g**2).sum() for g in grads)) for grads in per_rank_grads]
    assert float(local_norms[0]) > 10 * float(local_norms[1]), (
        "fixture must give the ranks very different scales for the point to be visible"
    )

    clipped_locally = []
    for grads, norm in zip(per_rank_grads, local_norms, strict=True):
        factor = min(1.0, 1.0 / (float(norm) + 1e-6))
        clipped_locally.append([g * factor for g in grads])
    local_factors = [min(1.0, 1.0 / (float(n) + 1e-6)) for n in local_norms]
    assert abs(local_factors[0] - local_factors[1]) > 0.5, (
        "local clipping should use visibly different multipliers per rank; if this now "
        "holds, the desync warning in the ADR can be revisited"
    )

    global_norm = torch.sqrt(sum((g**2).sum() for grads in per_rank_grads for g in grads))
    global_factor = min(1.0, 1.0 / (float(global_norm) + 1e-6))
    clipped_globally = [[g * global_factor for g in grads] for grads in per_rank_grads]
    ratios = [
        float(torch.sqrt(sum((g**2).sum() for g in grads))) / float(norm)
        for grads, norm in zip(clipped_globally, local_norms, strict=True)
    ]
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-5), (
        f"a global clip must scale every rank by the same ratio, got {ratios}"
    )
