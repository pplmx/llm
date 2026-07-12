# Refuse to start when `host != 127.0.0.1` and `api_key is None` (Finding N)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding N (MEDIUM — security)

## Description
`ServingConfig` defaults to `host="127.0.0.1"` (safe) and `api_key=None` (open). When
a user sets `LLM_SERVING_HOST=0.0.0.0` for a container, they may forget to set
`LLM_SERVING_API_KEY`. The current code accepts the request and returns 403 only when
the client provides a wrong key — there is no protection against anonymous requests.

## Acceptance criteria
- [ ] `main()` (or `lifespan`) raises `RuntimeError` at startup when `host` is not
      loopback and `api_key` is `None`
- [ ] Allow-list loopback: `127.0.0.0/8`, `::1`
- [ ] Add unit test covering: `127.0.0.1` + no key (OK), `0.0.0.0` + no key (raise),
      `0.0.0.0` + key (OK)
- [ ] Document the policy in `docs/guides/inference.md`

## Estimate
~30 minutes

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `security`, `serving`
