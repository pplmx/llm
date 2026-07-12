# Optimize `KVCache.update_at_indices` for non-1 `seq_len` (Finding AM)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding AM (MEDIUM),
Tier 2 #13

## Description
`src/llm/core/kv_cache.py:148-161` falls back to a Python-level loop
over batch slots when `seq_len_new > 1` (prefill). Each iteration calls
`.item()` (host-device sync) and a slice assignment. This stalls the
pipeline during prefill and contradicts the "in-place, no allocation"
promise of the cache. A pure tensor path using `scatter` (or
`index_put_` with `accumulate=False`) handles both decode (`seq_len=1`)
and prefill (`seq_len>1`) in one shot without host sync.

## Acceptance criteria
- [ ] `update_at_indices` implements a unified tensor path using
      `self.k_cache.index_put_` (or equivalent) when `start_pos` is a
      tensor of shape `[B_curr, seq_len_new]`; falls back to slice
      assignment only when `start_pos` is a Python int.
- [ ] No `.item()` call on the prefill path.
- [ ] Bench: `tests/perf/test_kv_cache_prefill.py` (marked `@pytest.mark.slow`)
      asserts the prefill path is at least 5× faster than the old loop on
      `B=8, S_new=128, max_seq_len=2048` (CPU is fine for this test).
- [ ] Existing tests (`tests/core/test_kv_cache.py` if present, plus
      any attention tests) pass unchanged.

## Estimate
~1 day

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `perf`, `core`, `serving`
