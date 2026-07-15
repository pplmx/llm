# 4. Paged Attention Serving Integration

Date: 2026-06-10

## Status

Accepted

## Context

`core/paged_attention/` provides block-level `PagedKVCache` and `paged_attention_forward`.
`ContinuousBatchingEngine` uses per-slot `KVCache` buffers with `batch_indices` for continuous batching.

## Decision

1. **ServingConfig flags** (`use_paged_attention`, `enable_prefix_cache`) wire into `ContinuousBatchingEngine.from_serving_config()`.
2. **`enable_prefix_cache`**: `SlotPrefixCache` reuses KV slots for identical prompt prefixes (implemented).
3. **`use_paged_attention`**: Instantiates `PagedKVCache` and routes the model
   forward through it (MHA / block / DecoderModel / engine all wired).
4. **`paged_kv_cache` parallel parameter**: MHA / TransformerBlock /
   DecoderModel accept a parallel `paged_kv_cache: PagedKVCache | None`
   argument alongside the existing linear `kv_cache`. When set, the
   linear cache is unused; the model writes K/V into the block
   allocator and reads via `paged_attention_forward`. The decoder
   iterates layers and passes `layer_idx=i` so MHA can slice the per-layer
   K/V tensor from `PagedKVCache.k_cache[i]`.
5. **`seq_id == slot_id`**: the engine's existing `SlotAllocator` doubles
   as the paged sequence identifier. Block tables live in
   `BlockManager` keyed by seq id, get cleared on `paged_kv_cache.free(seq_id)`
   when a request finishes.

## Consequences

**Advantages**:

- Prefix reuse reduces redundant prefill work today.
- Paged allocator + paged forward path are wired end-to-end; the
  memory benefit is now realised when `use_paged_attention=True`.
- The Python-fallback `paged_attention_forward` generalises to
  multi-token q (prefill + decode share the same kernel).
- `PagedKVCache.update` handles both fresh allocation and existing-sequence
  extension, so the same call site works for prefill and decode steps.

**Limitations**:

- The paged kernel is a Python gather / SDPA fallback; for production
  throughput, swap in a fused CUDA / Triton paged-attention kernel
  (e.g. `flash_attn_varlen_func` with block tables). Wiring is
  independent of that future optimisation.
- `attn_impl='mla'` supports the paged KV cache as of Tier 3 #31.
  The placeholder MLA caches K, V from `input_kv_proj` and runs the
  latent attention over the gathered cached context — the architectural
  benefit of per-position caching is limited (output is uniform-mean
  over latents) but the cache saves the `input_kv_proj` cost on
  incremental decode.
- `SlotPrefixCache` operates on dense slot buffers; the paged-side
  prefix-cache replay path (`PagedKVCache.add_prefix` +
  `try_get_prefix_blocks`) is not yet wired into the engine's
  `_lock_step_pre`. A follow-up slice.
- `flash_attn` rejects `paged_kv_cache` (no paged kernel exposed);
  use `attn_impl="mha"` when serving with paged KV.
