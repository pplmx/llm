# Cap `transformers<6` and `torch<3` upper bounds (Findings Q, R)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Findings Q (HIGH), R (MEDIUM)

## Description
`pyproject.toml` pins `transformers>=5.10.2` and `torch>=2.12.0` with no upper bound.
Both libraries evolve rapidly; a breaking change in 6.x or 3.x will break user
installs without warning. Cap until the project explicitly tests against the new major.

## Acceptance criteria
- [ ] `transformers>=5.10.2,<6.0.0` in `dependencies`
- [ ] `torch>=2.12.0,<3.0.0` in `dependencies`
- [ ] `uv lock` regenerated, lockfile committed
- [ ] Verify `make test` still passes locally

## Estimate
~10 minutes

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `dependencies`, `stability`
