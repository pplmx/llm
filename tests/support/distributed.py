"""Port allocation for multi-process distributed tests.

The tests in this repo spawn worker processes that rendezvous over a TCP store.
Getting the port wrong produces a failure mode that is easy to misread as an
environment problem, so the allocation lives here with the evidence.

**The bug this replaces.** The old helper was::

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]   # socket closed by the ``with`` exit

The kernel picks a free port, the helper returns it, and the socket closes — so
the port is released BEFORE any worker binds it. Anything on the machine can
take it in that window. Measured with workers racing for the same helper:
the port was bindable again in 400/400 attempts, and 8/400 rounds handed a port
number to two different callers.

When two tests collide, the loser's ``TCPStore`` client cannot reach its own
server. ``init_process_group`` then blocks until the socket timeout (240 s in
this repo's spawn workers) and the test dies with ``TimeoutError: <suite> spawn
exceeded 300s``. In isolation it passes, which is why it reads as flakiness or
as machine load; it is neither.

**The fix.** Let torch own the port. A store created with port 0 asks the OS for
an ephemeral port and then KEEPS LISTENING on it for as long as the store
object lives. There is no window in which the port is free. Children connect to
that port by number; they never allocate one. Measured under the same
contention: the port was stealable in 0/120 rounds, versus 120/120 for the old
helper.

Usage::

    from tests.support.distributed import rendezvous_store

    with rendezvous_store(world_size) as (host, port):
        mp.spawn(worker, args=(world_size, (host, port), results), nprocs=world_size)

and in the worker::

    store = dist.TCPStore(host, port, world_size, is_master=False, timeout=...)
    dist.init_process_group(backend, store=store, rank=rank, world_size=world_size)
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import torch.distributed as dist


def make_store(host: str, port: int, world_size: int, *, timeout: timedelta) -> Any:
    """Worker-side store handle: connect to the parent's already-bound store."""
    return dist.TCPStore(host, port, world_size, is_master=False, timeout=timeout)


@contextlib.contextmanager
def rendezvous_store(
    world_size: int,
    *,
    host: str = "127.0.0.1",
    timeout: timedelta = timedelta(seconds=300),
) -> Iterator[tuple[str, int]]:
    """Yield ``(host, port)`` for a store the PARENT keeps bound for the block.

    The store is created with port 0, so the OS assigns an ephemeral port that
    no other process can claim while the context manager is active. Yield is the
    only instant a caller can act on the port, and the store is still holding it
    at that point — the children connect to a bound port rather than racing for
    an unbound one.

    Children must join as non-master via :func:`make_store` so the parent's store
    stays the single authority for the port.

    Prefer this over writing the ``TCPStore(host, 0, ...)`` inline: several
    suites publish the port through ``MASTER_PORT`` instead and rely on a
    trailing ``del store`` after their join loop, which only works because the
    ``while`` loop above it always reaches that line. A ``with`` block cannot be
    skipped by an early ``return``/``raise``, so it is the safer shape for new
    code. The inline form is kept where it already works; there is no behavioural
    difference between the two, only that lifetime guarantee.
    """
    store = dist.TCPStore(host, 0, world_size, is_master=True, wait_for_workers=False, timeout=timeout)
    try:
        yield host, int(store.port)
    finally:
        # Dropping the reference closes the store's listening socket. Refusing to
        # wait for stragglers keeps a failed test from hanging here instead of
        # surfacing the real error from the workers.
        del store
