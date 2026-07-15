# Paged Attention full forward path (Tier 3 #3)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Tier 3 #3
("Paged Attention **full forward path** (replace `list[KVCache]`)"),
ROADMAP §10.2:
"Paged Attention — block allocator + prefix sidecar 已实现;
model forward 仍用 `list[KVCache]` ([ADR-004](../../adr/004-paged-attention-serving.md))"

## Description
The Paged Attention sidecar (`PagedKVCache`, `BlockManager`,
`BlockAllocator`, `paged_attention_forward`) is implemented and
unit-tested, but the serving path still feeds the model a regular
`KVCache` pool. As a result:

* `ServingConfig.use_paged_attention=True` raises
  `NotImplementedError` from `ContinuousBatchingEngine.from_serving_config`
  (Tier 1 #4 / Finding D).
* The model forward writes into linear `list[KVCache]` slots, never
  into the block-allocator storage. The sidecar is built but unused.
* Users can't actually get the memory benefit of Paged Attention in
  the continuous batching engine.

This ticket lands the full forward path as a foundation slice:

1. **`MultiHeadAttention.forward`** — accept an optional
   `paged_kv_cache: PagedKVCache | None` argument. When provided:
   * write the new K / V into the paged cache via
     `paged_kv_cache.update(seq_id, k, v)` (a per-sequence block write);
   * compute attention with `paged_attention_forward` instead of
     `sdpa` so we read K / V out of the block tables.
   Otherwise behaviour is unchanged (linear `KVCache` path).

2. **`TransformerBlock.forward`** — forward the new optional argument
   to its inner `MultiHeadAttention.forward` call. No new logic at this
   layer.

3. **`ContinuousBatchingEngine`** — replace the `NotImplementedError`
   raise with real wiring: when `use_paged_attention=True`, pass the
   `PagedKVCache` instance into the per-step model forward and manage
   per-request block tables. The existing linear `KVCache` pool is
   skipped (no double-allocation).

4. **Tests** — unit-test that `MHA.forward(paged_kv_cache=...)` writes
   into the block allocator and reads back via `paged_attention_forward`,
   producing the same attention output as the linear-cache path on the
   same input (within numerical tolerance). Plus a smoke test that
   `ContinuousBatchingEngine.from_serving_config(use_paged_attention=True)`
   no longer raises and that `engine.step()` produces output.

5. **ADR update** — flip ADR-004 status from "Accepted (partial)" to
   "Accepted" once the path is wired through; remove the "Limitations"
   section's first bullet.

## Acceptance criteria
- [ ] `MultiHeadAttention.forward` accepts `paged_kv_cache` and routes
      K / V through the block allocator + `paged_attention_forward`
      when it is provided.
- [ ] `TransformerBlock.forward` forwards `paged_kv_cache` unchanged.
- [ ] `ContinuousBatchingEngine.from_serving_config` no longer raises
      when `use_paged_attention=True`.
- [ ] `ContinuousBatchingEngine.step` produces correct output when
      `use_paged_attention=True` (smoke test against a tiny model).
- [ ] Tests cover the paged-cache write / read round-trip in MHA.
- [ ] ADR-004 status updated to "Accepted" with the limitation removed.
- [ ] ROADMAP §10.2 Paged Attention line marked complete.

## Out of scope (intentionally deferred)
- Fused CUDA kernels — the existing `paged_attention_forward` is a
  Python gather / SDPA fallback. Switching to a fused kernel
  (e.g. `vllm-flash-attn`) is a separate, larger slice.
- MLA + Paged Attention interaction — `attn_impl='mla'` is currently
  unsupported with any KV cache (see ADR-004 limitations). That is
  tracked under Tier 3 #5.
- Replacing the linear `KVCache` class with the paged cache across
  the training path — only the serving path switches in this slice.

## Estimate
~3 weeks for the foundation slice (no fused kernel).

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `paged-attention`, `serving`,
`core`, `attention`
