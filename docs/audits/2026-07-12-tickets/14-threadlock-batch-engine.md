# Add `threading.Lock` around `ContinuousBatchingEngine.step` (Finding AT)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding AT (HIGH)

## Description
`ContinuousBatchingEngine.step` mutates Python state (`self._seq_len`,
`self.free_slots`, `self.kv_caches`, prefix cache) without holding a lock. FastAPI's
`run_in_threadpool` calls `service.generate` from multiple threads, so concurrent
requests will race on these mutations. CUDA ops serialize internally but Python
bookkeeping does not.

## Acceptance criteria
- [ ] `ContinuousBatchingEngine.__init__` allocates `self._step_lock = threading.Lock()`
- [ ] `step()` body is wrapped `with self._step_lock:` — only the Python-bookkeeping
      portion, NOT the inner model forward (which we want to release during async
      eventually)
- [ ] Add a stress test: spawn 8 threads, each calling `add_request` + `step()` 50
      times; assert no crashes and scheduler invariants hold
- [ ] Document the locking policy in `docs/guides/inference.md`

## Estimate
~45 minutes

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `correctness`, `serving`, `concurrency`
