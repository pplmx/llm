# Structured APIError envelope + request IDs (Finding K)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding K (MEDIUM),
Tier 2 #2

## Description
`serving/api.py` raises ad-hoc `HTTPException(detail=...)` with no envelope,
no request-ID correlation, and no machine-readable error code. Operators
triaging a 500 cannot grep one request across uvicorn access log + app log
+ client trace. Clients cannot programmatically branch on a stable code.

## Acceptance criteria
- [ ] New module `src/llm/serving/errors.py` exporting `APIError`,
      `ErrorCode` (enum), and a `to_envelope(exc, request_id) -> dict`
      helper.
- [ ] Envelope shape:
      `{"error": {"code": "<stable_id>", "message": "<human>",
      "details": {...}, "request_id": "<uuid>"}}`.
- [ ] `FastAPI` exception handler (registered in `api.py` lifespan or app
      factory) converts `APIError` and `HTTPException` to the envelope.
      `RequestValidationError` from pydantic also routes through it.
- [ ] ASGI middleware assigns `X-Request-ID` (generate if absent, honor
      inbound header) and stores it on `request.state.request_id`. The
      middleware also logs `request_id`, `method`, `path`, `status`,
      `duration_ms` at INFO level on response.
- [ ] Response includes `X-Request-ID` echo header.
- [ ] All existing endpoints (`/generate`, `/batch_generate`,
      `/v1/chat/completions`) raise `APIError` instead of bare
      `HTTPException`. Error codes: `INVALID_REQUEST`, `TIMEOUT`,
      `MODEL_UNAVAILABLE`, `INTERNAL`, `UNAUTHORIZED`.
- [ ] Tests: `tests/serving/test_errors.py` covers envelope shape,
      request-ID propagation, validation error mapping, and that
      `UNAUTHORIZED` is raised by `get_api_key` when api_key mismatch.
- [ ] Tests: `tests/serving/test_request_id.py` covers middleware echo,
      generation when absent, and that downstream logs include it.

## Estimate
~1.5 days

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `api`, `serving`, `observability`
