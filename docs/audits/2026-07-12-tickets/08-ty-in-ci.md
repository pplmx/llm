# Add `ty` type check to CI lint job (Finding W)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding W (MEDIUM)

## Description
`Makefile:58` defines `make ty` (runs `uvx ty check`), but `.github/workflows/ci.yml`
only runs `ruff format --check` and `ruff check`. Type-check drift will not be caught.

## Acceptance criteria
- [x] `ci.yml` `lint` job adds `uvx ty check src/llm/` after the ruff steps
- [x] No new failures introduced (all pre-existing `ty` errors in `src/llm/` fixed)
- [x] `ty` failure blocks PR merge (`continue-on-error` removed 2026-08-08)

## Resolution
The type-check scope is `src/llm/` (matching the package layout and CI), not the
whole repo: `_learning/`, `notebooks/`, `scripts/` and `tests/` contain
experimental/legacy code and are intentionally excluded from the gate. The
`Makefile` `ty` target was aligned to the same scope.

## Estimate
~10 minutes (plus any pre-existing fixes)

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `ci`, `quality`
