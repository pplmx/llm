# Log model version + config on startup (Finding AZ)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding AZ (HIGH)

## Description
`lifespan` in `src/llm/serving/api.py` initializes the engine but operators have no
way to verify from logs what model is being served, what dtype, what attention impl,
or whether prefix cache is on. This makes incident triage impossible.

## Acceptance criteria
- [ ] `lifespan` (or `ServingGenerationService.from_config`) logs a single structured
      JSON line at startup containing: model class, param count, dtype, device,
      `max_seq_len`, `attn_impl`, `mlp_impl`, `generation_backend`,
      `enable_prefix_cache`, `use_paged_attention`, `api_key_set` (bool only)
- [ ] Add unit test asserting the log line is emitted with required keys
- [ ] Add `docs/guides/inference.md` example showing how to grep for it

## Estimate
~20 minutes

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `observability`, `serving`
