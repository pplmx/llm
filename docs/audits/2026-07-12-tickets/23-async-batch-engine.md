# Make `ContinuousBatchingEngine.step` truly async (Finding M)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding M (HIGH),
Tier 2 #1

## Description
`ContinuousBatchingEngine.step` runs entirely synchronously inside
`run_in_threadpool`. With the `threading.Lock` guard (ticket 14) added,
that becomes a serious bottleneck: only one request can advance through
the scheduler per worker thread. The right model is:
- Python bookkeeping (slot alloc, KV slice) under lock.
- Model forward / sampling released to the GPU without the lock.
- Wake-up event when results land (asyncio.Event) so the API layer can
  yield to other requests.

This ticket establishes the foundation: split `step` into `_lock_step`
(Python bookkeeping) + `_forward_and_sample` (compute, lock released),
and provide an `async def step()` wrapper. Wiring all endpoints to the
async path is a separate follow-up.

## Acceptance criteria
- [ ] `ContinuousBatchingEngine._lock_step()` contains only the Python
      bookkeeping that needs the lock (slot alloc/free, request queue
      dequeue, prefix-cache lookup), under `with self._step_lock`.
- [ ] `ContinuousBatchingEngine._forward_and_sample()` runs the model
      forward and sampling **without** holding `self._step_lock`.
- [ ] New `async def step_async()` that: acquires the lock for the
      bookkeeping portion, releases it, awaits
      `asyncio.to_thread(self._forward_and_sample)`, then re-acquires
      the lock for the post-compute bookkeeping (free slots, set
      outputs).
- [ ] `step()` (sync) kept as the thin composition:
      `self._lock_step(); self._forward_and_sample(); self._lock_step_post()`.
- [ ] Existing concurrency stress test (ticket 14) still passes.
- [ ] New micro-benchmark in `tests/perf/test_batch_engine_async.py`
      (marked slow) shows `step_async` saturates ≥ N concurrent
      in-flight requests vs. `step` (N=4 on CPU is sufficient).
- [ ] Doc: `docs/guides/inference.md` section on the new lifecycle.

## Estimate
~1.5 weeks

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `perf`, `serving`,
`concurrency`, `correctness`
