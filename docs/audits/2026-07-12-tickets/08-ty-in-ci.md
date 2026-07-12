# Add `ty` type check to CI lint job (Finding W)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding W (MEDIUM)

## Description
`Makefile:58` defines `make ty` (runs `uvx ty check`), but `.github/workflows/ci.yml`
only runs `ruff format --check` and `ruff check`. Type-check drift will not be caught.

## Acceptance criteria
- [ ] `ci.yml` `lint` job adds `uvx ty check .` after the ruff steps
- [ ] No new failures introduced (fix any pre-existing `ty` errors first)
- [ ] `ty` failure blocks PR merge

## Estimate
~10 minutes (plus any pre-existing fixes)

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `ci`, `quality`
