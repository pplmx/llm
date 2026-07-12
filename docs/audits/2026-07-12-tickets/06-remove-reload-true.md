# Remove `reload=True` from `llm-serve` entry (Finding O)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding O (LOW)

## Description
`src/llm/serving/api.py:405`: `uvicorn.run("llm.serving.api:app", host=config.host,
port=8000, reload=True)`. `reload=True` enables uvicorn's file-watcher, which conflicts
with `from llm.serving.api import app` (the watch import path is incompatible with
production). It also silently restarts the server on any `.py` change.

## Acceptance criteria
- [ ] `main()` removes `reload=True`
- [ ] `reload=True` is opt-in via a `LLM_SERVING_RELOAD=true` env var for dev workflows
- [ ] Update docs to show `make dev` or equivalent for local dev with reload

## Estimate
~5 minutes

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `serving`, `good-first-issue`
