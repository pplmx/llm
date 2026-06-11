# 4. Paged Attention Serving Integration

Date: 2026-06-10

## Status

Accepted (partial)

## Context

`core/paged_attention/` provides block-level `PagedKVCache` and `paged_attention_forward`.
`ContinuousBatchingEngine` uses per-slot `KVCache` buffers with `batch_indices` for continuous batching.

## Decision

1. **ServingConfig flags** (`use_paged_attention`, `enable_prefix_cache`) wire into `ContinuousBatchingEngine.from_serving_config()`.
2. **`enable_prefix_cache`**: `SlotPrefixCache` reuses KV slots for identical prompt prefixes (implemented).
3. **`use_paged_attention`**: Instantiates `PagedKVCache` as a block-allocator sidecar for prefix metadata and future kernel integration.
4. **DecoderModel forward** continues to use `list[KVCache]` until MHA gains a paged forward path.

## Consequences

**Advantages**:

- Prefix reuse reduces redundant prefill work today.
- Paged allocator is ready for full kernel wiring without API churn.

**Limitations**:

- `use_paged_attention=True` does not yet switch the model forward to `paged_attention_forward`.
- `attn_impl='mla'` does not support KV cache (latent attention architecture).
