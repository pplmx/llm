# Split heavy runtime deps into `[eval]` and `[logging]` groups (Finding P)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding P (MEDIUM)

## Description
`pyproject.toml` runtime deps include `lm_eval`, `scikit-learn`, `seaborn`,
`rouge-score`, `sacrebleu`, and `tensorboard` — all ~100MB+ collectively. None of
them are required for core training or serving. They should be opt-in.

## Acceptance criteria
- [ ] New groups `[eval]` (rouge, sacrebleu, scikit-learn, seaborn, lm_eval) and
      `[logging]` (tensorboard) added to `[dependency-groups]` AND
      `[project.optional-dependencies]`
- [ ] These packages removed from runtime `dependencies`
- [ ] Existing evaluation tests pass under `uv sync --group eval`
- [ ] Existing training tests pass with the smaller default install
- [ ] `pyproject.toml` comment explains the split

## Estimate
~1 hour

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `dependencies`, `good-first-issue`
