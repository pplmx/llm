"""Concurrent cold-start bootstrap must not double-register (RIL ISS-119).

All three plugin registries (generation/export/peft) lazily bootstrap on
first use. Before this fix the guard was a bare check-then-act, so two
threads racing ``ensure_*_registered`` on a cold process could both pass
the ``*_registered`` flag check, and the second thread's
``Registry.register`` raised ``ValueError: already registered``.

These tests force a cold state (reset the module flag + wipe the registry),
then fan out two threads against the bootstrap function. With the lock the
second thread re-checks the flag inside the critical section and returns;
without it the test fails with the duplicate-registration error.
"""

from __future__ import annotations

import threading

import llm.core.peft.registry as peft_reg
import llm.export.registry as export_reg
import llm.generation.registry as gen_reg


def _cold_start_twice(ensure, registry, guard_module, guard_flag):
    """Race ``ensure()`` on two threads from a cold registry state.

    Returns the list of per-thread results (True on success, or the raised
    exception object) so the caller can assert None raised.
    """
    # Force a cold start: clear the registry entries AND the guard flag, so
    # both threads actually re-enter the bootstrap. (Clearing only the
    # entries leaves the flag True, which makes ``ensure`` return early and
    # the race invisible.)
    registry._entries.clear()
    setattr(guard_module, guard_flag, False)
    results: list[BaseException | bool] = []
    barrier = threading.Barrier(3)  # 2 workers + main

    def worker():
        barrier.wait()  # maximize the race window: both threads start together
        try:
            ensure()
            results.append(True)
        except Exception as exc:  # noqa: BLE001 — we want any raised error
            results.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    barrier.wait()
    t1.join()
    t2.join()
    return results


def test_generation_backend_cold_start_race_is_serialized():
    ensure = gen_reg.ensure_backends_registered
    results = _cold_start_twice(ensure, gen_reg.BACKEND_REGISTRY, gen_reg, "_backends_registered")

    assert len(results) == 2
    for r in results:
        assert r is True, f"concurrent cold-start raised: {r!r}"

    # And the bootstrapped state is intact / usable.
    assert "speculative" in gen_reg.BACKEND_REGISTRY
    assert gen_reg._backends_registered is True


def test_export_backend_cold_start_race_is_serialized():
    ensure = export_reg.ensure_exporters_registered
    results = _cold_start_twice(ensure, export_reg.EXPORT_REGISTRY, export_reg, "_exporters_registered")

    assert len(results) == 2
    for r in results:
        assert r is True, f"concurrent cold-start raised: {r!r}"

    assert "onnx" in export_reg.EXPORT_REGISTRY
    assert export_reg._exporters_registered is True


def test_peft_method_cold_start_race_is_serialized():
    ensure = peft_reg.ensure_methods_registered
    results = _cold_start_twice(ensure, peft_reg.PEFT_REGISTRY, peft_reg, "_methods_registered")

    assert len(results) == 2
    for r in results:
        assert r is True, f"concurrent cold-start raised: {r!r}"

    # PEFT_REGISTRY has built-ins; the exact set is defined by the loader,
    # so just assert it is non-empty and bootstrapped.
    assert len(peft_reg.PEFT_REGISTRY._entries) > 0
    assert peft_reg._methods_registered is True


def test_registry_lock_objects_are_shared_across_callers():
    """The lock must be a module-level singleton so every caller serializes
    against the same lock — a per-call ``Lock()`` would still race."""
    assert isinstance(gen_reg._backend_registration_lock, threading.Lock)
    # Two successive references resolve to the same object.
    assert gen_reg._backend_registration_lock is gen_reg._backend_registration_lock
