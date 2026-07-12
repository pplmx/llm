# Hypothesis invariant tests for core invariants (Finding AD)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding AD (MEDIUM),
Tier 2 #8

## Description
Existing tests are mostly example-based — they verify specific (input,
expected output) pairs. Properties of the registry kernel, KV cache,
generation sampling, and norm factory have never been fuzzed. Hypothesis
finds the corner cases example-based tests miss (empty lists, duplicate
keys, off-by-one seq_len, NaN logits, etc.). Adding 5–10 property-based
tests will harden the parts of the codebase that are hardest to debug
when they break at scale.

## Acceptance criteria
- [ ] Add `hypothesis>=6.0` as a dev dependency in `pyproject.toml`.
- [ ] New test module per invariant:
  - `tests/core/test_registry_invariants.py` — registering the same name
    twice raises; `get` returns the same instance; iteration order is
    insertion order; empty registry is iterable.
  - `tests/core/test_kv_cache_invariants.py` — after N updates with
    random seq_len, cached slice matches a naive `torch.cat` reference;
    overflow raises `ValueError`; `reset` restores `_seq_len=0`.
  - `tests/core/test_norm_factory_invariants.py` — every registered
    norm accepts `hidden_size` and `eps`; output shape matches input
    shape; factory returns a *new* instance per call (no shared state).
  - `tests/generation/test_sampling_invariants.py` — sampling
    probabilities sum to 1 within `1e-5`; top-k restricts to k;
    repetition penalty is monotonic in token count.
- [ ] Each test runs in ≤2s with the default Hypothesis profile; mark
      with `@settings(max_examples=50, deadline=None)` where needed.
- [ ] CI passes; no skipped tests.

## Estimate
~1 day

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `testing`, `quality`
