# Add `attn_impl` × `use_cache` validator in ModelConfig (Finding E)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding E (MEDIUM)

## Description
`attn_impl="mla"` does not support KV cache, but the failure currently surfaces deep in
the attention layer's forward pass during training. Fail fast in config validation.

## Acceptance criteria
- [ ] `ModelConfig.check_consistency` adds a validator that raises `ValueError` when
      `attn_impl == "mla"` and a downstream consumer would request KV cache. Since
      `use_cache` is not in `ModelConfig`, the validator should consult a static
      capability map (`attn_impl_supports_kv_cache: dict[str, bool]`) co-located with
      the registry declaration
- [ ] Add unit test for the validator
- [ ] Update `docs/reference/architecture.md` to mention the capability map

## Estimate
~30 minutes

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `config`, `fail-fast`
