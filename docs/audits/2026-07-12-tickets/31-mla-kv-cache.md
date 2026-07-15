# MLA + KV cache (Tier 3 #5)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Tier 3 #5
("MLA + KV cache (full implementation)"),
ROADMAP §10.2 (Paged Attention is now full forward — this is the next
sibling work).

## Description
``MultiLatentAttention`` (``attn_impl='mla'``) currently raises
``ValueError("MultiLatentAttention does not support KV cache.")`` whenever
``use_cache=True`` or ``kv_cache`` is supplied. The capability map
``set_attention_kv_cache_capability("mla", supports=False)`` reflects
this, and ``ModelConfig.check_consistency`` rejects the combination.

The placeholder MLA in this repo is **not** DeepSeek-V2-style MLA
(there is no latent-compression of K, V, no decoupled RoPE). It is
a latent cross-attention block: learnable latent queries attend to
``input_kv_proj(x)``, and the output is a uniform average over the
latents broadcast to every position. That uniformity limits the
architectural value of per-position KV caching — but the framework
can still cache ``K, V`` to avoid re-projecting past tokens on
incremental decode.

This slice is the foundation: wire the existing MLA placeholder into
the standard ``KVCache`` + ``PagedKVCache`` pool so the engine and
training path can run ``attn_impl='mla'`` end-to-end. Real
DeepSeek-V2-style MLA (latent-dim compression + decoupled RoPE) is a
larger follow-up that this slice unblocks.

1. ``MultiLatentAttention.forward`` — accept ``kv_cache`` and
   ``paged_kv_cache`` like MHA. Cache the per-token K, V from
   ``input_kv_proj``. On decode, project just the new token, concat
   with cached K, V, then run the latent attention. The uniform-mean
   output remains; only the K, V source changes.

2. ``set_attention_kv_cache_capability("mla", supports=True)`` — flip
   the capability flag so ``ModelConfig.check_consistency`` accepts
   ``attn_impl='mla'`` with KV cache. Document the architectural caveat
   (uniform output) in the docstring.

3. ``FlashAttention`` already accepts ``paged_kv_cache`` (and rejects
   it explicitly with a helpful error) — no change.

4. ``TransformerBlock`` + ``DecoderModel`` — verify the existing
   ``paged_kv_cache`` + ``layer_idx`` plumbing threads through to MLA
   correctly (no Union dispatch needed; the parallel-parameter pattern
   added in T3 #3 already supports this).

5. ``ContinuousBatchingEngine`` — no code change. With ``attn_impl='mla'``
   + ``use_paged_attention=True``, the existing path forwards
   ``paged_kv_cache`` to the model, which dispatches inside MLA's
   forward. Add a smoke test that a tiny MLA model runs end-to-end
   through the engine.

## Acceptance criteria
- [ ] ``MultiLatentAttention.forward`` accepts ``kv_cache`` /
      ``paged_kv_cache`` + ``layer_idx`` and routes K, V through the
      configured cache.
- [ ] ``set_attention_kv_cache_capability("mla", supports=True)``.
- [ ] MLA forward + KV cache produces the same output as a full-sequence
      MLA forward within ``atol=1e-5`` (incremental decode equivalence).
- [ ] Engine smoke test: a 1-layer MLA model runs ``step()`` end-to-end.
- [ ] Docstring on ``MultiLatentAttention`` documents the placeholder
      nature and the architectural caveat (uniform output).

## Out of scope (intentional)
- DeepSeek-V2-style compression (latent KV dim decoupling, decoupled
  RoPE, absorbed Q-K transposed into the latent). That is a real
  architectural rewrite — tracked as a follow-up.
- Per-token compression. The current placeholder caches full K, V
  tensors like MHA does; the block-allocator integration just works.

## Estimate
~1 week for the foundation slice (no architectural rewrite).

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `mla`, `kv-cache`, `core`,
`attention`
