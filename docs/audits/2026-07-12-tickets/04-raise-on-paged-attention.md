# Raise on `use_paged_attention=True` until full path implemented (Finding D)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding D (HIGH)

## Description
`docs/adr/004-paged-attention-serving.md` and `ROADMAP.md` both document that Paged
Attention is **partial**: the sidecar exists and prefix cache works, but model forward
still uses `list[KVCache]`. Users enabling `use_paged_attention=True` in `ServingConfig`
get no benefit and may not realize it. Make the partial state explicit.

## Acceptance criteria
- [ ] `ContinuousBatchingEngine.__init__` raises `NotImplementedError` when
      `use_paged_attention=True` and the engine is built via `from_serving_config`
      (i.e., for serving)
- [ ] Direct construction (with explicit kwargs in tests) is unchanged
- [ ] Error message points to ADR-004 and the planned milestone
- [ ] Add unit test for the raise path

## Estimate
~15 minutes

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `serving`, `documentation`
