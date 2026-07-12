# Use hmac.compare_digest for API key check (Finding AS)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding AS (HIGH — timing attack)

## Description
`src/llm/serving/api.py:71, 76` compares the API key with `==`. `==` on strings is not
constant-time and leaks key bytes via timing. Use `hmac.compare_digest`.

## Acceptance criteria
- [ ] Replace both `==` comparisons with `hmac.compare_digest` in `get_api_key`
- [ ] Add unit test covering: matching key, mismatching key, empty header, wrong scheme
- [ ] Test should NOT assert timing properties (those are flaky); just correctness

## Estimate
~5 minutes

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `security`
