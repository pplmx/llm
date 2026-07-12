# Custom Prometheus metrics for serving (Finding AX)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding AX (MEDIUM),
Tier 2 #12

## Description
`prometheus-fastapi-instrumentator` gives generic HTTP RED metrics
(request rate, error rate, duration) but no domain metrics: tokens
generated per request, batch fill ratio, KV-cache hit rate, queue depth.
Operators running a serving fleet cannot observe what they actually care
about without a sidecar scraper.

## Acceptance criteria
- [ ] New module `src/llm/serving/metrics.py` defining a `ServingMetrics`
      dataclass with the following `prometheus_client` counters /
      histograms / gauges:
  - `llm_tokens_generated_total` (Counter, labels: `endpoint`)
  - `llm_tokens_per_request` (Histogram, labels: `endpoint`, buckets:
    16, 64, 256, 1024, 4096)
  - `llm_request_duration_seconds` (Histogram, labels: `endpoint`,
    `status`, buckets: 0.05, 0.25, 1, 5, 30)
  - `llm_batch_fill_ratio` (Gauge, range 0–1, updated per `step()`)
  - `llm_kv_cache_hit_ratio` (Gauge, range 0–1, updated per request
    completion)
  - `llm_inflight_requests` (Gauge)
- [ ] Wire into `serving/api.py`:
  - `llm_inflight_requests` `inc()`/`dec()` around the
    `inference_semaphore`.
  - `llm_tokens_per_request` `.observe()` after each successful
    generate.
  - `llm_batch_fill_ratio` updated from `engine.step()` return value
    (extend `ContinuousBatchingEngine.step` to return `(scheduled,
    total_active_slots)` or equivalent — without breaking the existing
    return contract).
- [ ] Document the metrics in `docs/guides/inference.md` with example
      PromQL queries (p95 latency, batch utilization, cache hit).
- [ ] Test: `tests/serving/test_metrics.py` asserts metrics are emitted
      on a minimal request flow.

## Estimate
~1 day

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `observability`, `serving`
