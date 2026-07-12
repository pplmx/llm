# Add `coverage --fail-under` gate to CI (Finding AA)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding AA (MEDIUM)

## Description
`ci.yml` runs `make test-cov` and uploads the report as an artifact, but does not fail
if coverage drops. The 80%+ target documented in `CONTRIBUTING.md` is not enforced.

## Acceptance criteria
- [ ] CI test step adds `--cov-fail-under=80` (or current measured value)
- [ ] Find the actual current coverage first via `make test-cov` locally
- [ ] If current coverage is < 80%, set the gate at the current value and open
      follow-up issues for the gap (do not silently lower the bar)
- [ ] Document the policy in `CONTRIBUTING.md`

## Estimate
~15 minutes

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `ci`, `quality`, `testing`
