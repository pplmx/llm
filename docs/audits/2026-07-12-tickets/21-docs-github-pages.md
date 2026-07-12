# Deploy docs to GitHub Pages (Finding AI)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding AI (MEDIUM),
Tier 2 #10

## Description
`mkdocs.yml` is configured but `.github/workflows/docs.yml` either
doesn't exist or doesn't publish. The audit found that "docs exist but
are unreachable" — a real discoverability loss.

## Acceptance criteria
- [ ] `.github/workflows/docs.yml` exists, triggered on push to `main`
      when `docs/**`, `mkdocs.yml`, or `mkdocs.yml.lock` change.
- [ ] Workflow steps: `actions/checkout` (depth: 0 for git-revision-date
      plugin), `actions/setup-python` (Python 3.14), `pip install
      mkdocs mkdocs-material mkdocstrings[python]`, `mkdocs build
      --strict`, `actions/upload-pages-artifact`,
      `actions/deploy-pages` (environment: `github-pages`).
- [ ] `mkdocs build --strict` succeeds locally (the build is the source
      of truth for whether mkdocstrings ticket is done).
- [ ] Add `docs/CNAME` placeholder with a comment noting the user must
      add their domain.
- [ ] README badge: `[![Docs](...)](https://<owner>.github.io/llm/)`.

## Estimate
~2 hours

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `docs`, `ci`, `discoverability`
