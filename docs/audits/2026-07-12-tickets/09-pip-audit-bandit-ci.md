# Add `pip-audit` and `bandit` to CI (Finding X)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding X (MEDIUM)

## Description
No `pip-audit`, `bandit`, `safety`, or `trivy` runs in CI. For a project that
downloads model weights and serves them over HTTP, this is non-trivial. Add two
gates: dependency vulnerability scan and static security analysis.

## Acceptance criteria
- [ ] Add `security` job to `ci.yml` running `uvx pip-audit` and `uvx bandit -r src/`
- [ ] `pip-audit` runs on a weekly schedule (cron) AND on every PR
- [ ] `bandit` runs on every PR; findings of severity `high` fail the build
- [ ] Pin tool versions in `pyproject.toml` `[dependency-groups]`
- [ ] Document findings policy in `CONTRIBUTING.md`

## Estimate
~45 minutes

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `security`, `ci`
