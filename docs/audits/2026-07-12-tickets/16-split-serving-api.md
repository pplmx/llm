# Split `serving/api.py` into focused modules (Finding H)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding H (MEDIUM),
Tier 2 #6

## Description
`src/llm/serving/api.py` is 512 LOC mixing: lifespan + startup config
logging, auth, two text endpoints, OpenAI-compatible chat completions
(non-streaming + streaming), `_messages_to_prompt`, request validation,
error handling, Prometheus instrumentator wiring, and the `main()`
CLI entrypoint. Each is small on its own but together they obscure
where to look for any change. Splitting makes the boundary between
"transport" and "presentation" visible.

## Acceptance criteria
- [ ] New modules:
  - `src/llm/serving/auth.py` — `get_api_key`, `_extract_bearer_token`,
    `_is_loopback` (currently in `api.py`).
  - `src/llm/serving/errors.py` — see ticket 15 (depends-on).
  - `src/llm/serving/middleware.py` — `RequestIDMiddleware`.
  - `src/llm/serving/routers/health.py`, `routers/generate.py`,
    `routers/chat.py` — one `APIRouter` per logical endpoint group.
  - `src/llm/serving/chat_template.py` — `_messages_to_prompt`,
    `DEFAULT_CHAT_MESSAGE_TEMPLATE`,
    `DEFAULT_CHAT_GENERATION_PREFIX`.
  - `src/llm/serving/cli.py` — `main()` and `_is_loopback` public-host
    guard.
- [ ] `api.py` shrinks to ≤120 LOC and contains only: import + `app =
      FastAPI(...)` + `lifespan` + `Instrumentator().instrument(app)` +
      `include_router(...)` calls + `if __name__ == "__main__"`.
- [ ] No behavioral change: all existing tests pass unchanged.
- [ ] Add a smoke test in `tests/serving/test_imports.py` confirming all
      new modules import cleanly and `app` is built without warnings.

## Estimate
~3 hours

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `refactor`, `serving`,
`maintainability`
