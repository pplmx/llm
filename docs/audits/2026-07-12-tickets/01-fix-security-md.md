# Fix SECURITY.md (Finding AQ)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding AQ (HIGH)

## Description
`SECURITY.md` lists supported versions as `5.1.x`, `5.0.x`, `4.0.x` — none match the
project (currently v0.0.5). This is a copy-paste artifact and undermines the project's
security posture in the eyes of reporters and downstream users.

## Acceptance criteria
- [ ] Version table replaced with actual supported versions (0.0.x line)
- [ ] `## Reporting a Vulnerability` section includes: contact channel, expected
      response window, disclosure policy
- [ ] Markdown passes `rumdl` lint

## Estimate
~10 minutes

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `security`, `good-first-issue`
