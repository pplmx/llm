"""Tests for :mod:`llm.serving.metrics` (T2 #22, Finding AX).

Covers the structure of the ``ServingMetrics`` collection (counters,
histograms, gauges, labels, buckets) and the convenience helpers used
by the serving tier to record observations. Uses a private
``CollectorRegistry`` per test so we never collide with the default
Prometheus registry used by ``/metrics`` in production.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from llm.serving.metrics import METRICS, ServingMetrics


def _fresh() -> ServingMetrics:
    """Build a ``ServingMetrics`` with an isolated registry."""
    return ServingMetrics(registry=CollectorRegistry())


# --- Shape of the metric collection -----------------------------------------


def test_serving_metrics_defines_all_required_metrics():
    """All required Prometheus metrics exist with the correct types."""
    m = _fresh()
    assert isinstance(m.tokens_generated_total, Counter)
    assert isinstance(m.tokens_per_request, Histogram)
    assert isinstance(m.request_duration_seconds, Histogram)
    assert isinstance(m.batch_fill_ratio, Gauge)
    assert isinstance(m.kv_cache_hit_ratio, Gauge)
    assert isinstance(m.inflight_requests, Gauge)


def test_serving_metrics_labels():
    """Label sets match the ticket acceptance criteria."""
    m = _fresh()
    assert "endpoint" in m.tokens_generated_total._labelnames
    assert "endpoint" in m.tokens_per_request._labelnames
    assert "endpoint" in m.request_duration_seconds._labelnames
    assert "status" in m.request_duration_seconds._labelnames


def test_serving_metrics_buckets():
    """Histogram buckets match the ticket acceptance criteria.

    Prometheus automatically appends a ``+Inf`` bucket to every histogram;
    we ignore it and compare the configured finite buckets only.
    """
    m = _fresh()
    tpr_buckets = list(m.tokens_per_request._upper_bounds)  # type: ignore[attr-defined]
    assert tpr_buckets[:-1] == [16.0, 64.0, 256.0, 1024.0, 4096.0]

    dur_buckets = list(m.request_duration_seconds._upper_bounds)  # type: ignore[attr-defined]
    assert dur_buckets[:-1] == [0.05, 0.25, 1.0, 5.0, 30.0]


def test_module_level_singleton_uses_default_registry():
    """The module-level ``METRICS`` singleton is wired into the default registry.

    ``Instrumentator.expose(app)`` only emits metrics in the default
    registry, so the singleton MUST live there for ``/metrics`` to show
    these counters.
    """
    from prometheus_client import REGISTRY

    assert METRICS.registry is REGISTRY


# --- Convenience helpers ----------------------------------------------------


def test_observe_tokens_increments_counter_and_histogram():
    """observe_tokens adds to both the total counter and the per-request histogram."""
    m = _fresh()
    m.observe_tokens(endpoint="generate", token_count=42)

    counter_value = m.tokens_generated_total.labels(endpoint="generate")._value.get()
    assert counter_value == 42

    # Histogram: total observation count and sum reflect the single observation.
    count, total = _histogram_count_and_sum(m.tokens_per_request, endpoint="generate")
    assert count == 1
    assert total == 42


def test_inflight_context_manager_inc_dec():
    """The inflight gauge increments on enter and decrements on exit."""
    m = _fresh()
    assert m.inflight_requests._value.get() == 0  # type: ignore[attr-defined]

    with m.track_inflight():
        assert m.inflight_requests._value.get() == 1  # type: ignore[attr-defined]
        with m.track_inflight():
            assert m.inflight_requests._value.get() == 2  # type: ignore[attr-defined]
        assert m.inflight_requests._value.get() == 1  # type: ignore[attr-defined]
    assert m.inflight_requests._value.get() == 0  # type: ignore[attr-defined]


def test_inflight_decrements_on_exception():
    """Exceptions inside the context still release the inflight slot."""
    m = _fresh()
    try:
        with m.track_inflight():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert m.inflight_requests._value.get() == 0  # type: ignore[attr-defined]


def test_record_batch_fill_ratio_clamps_to_unit_interval():
    """record_batch_fill_ratio rejects out-of-range values."""
    import pytest

    m = _fresh()
    m.record_batch_fill_ratio(scheduled=4, total_active_slots=8)
    assert m.batch_fill_ratio._value.get() == 0.5  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="ratio"):
        m.record_batch_fill_ratio(scheduled=-1, total_active_slots=4)
    with pytest.raises(ValueError, match="ratio"):
        m.record_batch_fill_ratio(scheduled=5, total_active_slots=4)
    with pytest.raises(ValueError, match="total_active_slots"):
        m.record_batch_fill_ratio(scheduled=0, total_active_slots=0)


def test_record_kv_cache_hit_ratio_clamps_to_unit_interval():
    """record_kv_cache_hit_ratio accepts floats in [0, 1] and rejects others."""
    import pytest

    m = _fresh()
    m.record_kv_cache_hit_ratio(0.75)
    assert m.kv_cache_hit_ratio._value.get() == 0.75  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="ratio"):
        m.record_kv_cache_hit_ratio(-0.1)
    with pytest.raises(ValueError, match="ratio"):
        m.record_kv_cache_hit_ratio(1.5)


def test_request_timer_records_duration_and_status():
    """request_timer yields a context manager that records latency and status."""
    import time

    m = _fresh()
    with m.request_timer(endpoint="generate") as timer:
        time.sleep(0.01)
        timer.set_status(200)

    count, total = _histogram_count_and_sum(m.request_duration_seconds, endpoint="generate", status="200")
    assert count == 1
    assert total >= 0.01


def _histogram_count_and_sum(hist: Histogram, **labels: str) -> tuple[int, float]:
    """Return ``(_count, _sum)`` for a labelled histogram by walking samples.

    ``prometheus_client`` does not expose ``_count`` / ``_sum`` as direct
    attributes on the labelled child metric, so we read them via
    ``.collect()`` (the same path the ``/metrics`` endpoint uses).
    """
    labelled = hist.labels(**labels)
    for metric in labelled.collect():
        for sample in metric.samples:
            if sample.name.endswith("_count"):
                count = int(sample.value)
            elif sample.name.endswith("_sum"):
                total = float(sample.value)
    return count, total


def test_generate_stream_counts_decoded_tokens_not_chunks(monkeypatch):
    """A multi-token chunk (the stop-buffer drains several tokens as ONE
    chunk) must count every DECODED token, not 1 per chunk.

    Regression: ``/generate?stream=true`` did ``token_count += 1`` per
    received chunk, but with ``stop`` sequences ``stream_generate`` yields
    several tokens through its stop buffer as a single chunk — the usage
    metric undercounted and the chat route's ``finish_reason``
    (``"length" if token_count >= max_tokens``) mis-reported stop-vs-length.
    """
    import asyncio
    from types import SimpleNamespace

    import llm.serving.routers.generate as gen
    from llm.serving.schemas import GenerationRequest

    m = _fresh()
    monkeypatch.setattr(gen, "metrics", m)
    monkeypatch.setattr(gen, "config", SimpleNamespace(request_timeout=10.0))
    monkeypatch.setattr(gen, "inference_semaphore", None)  # -> _null_cm()
    # Char-level serving tokenizer: 1 id per char, so _token_count("abc") == 3.
    monkeypatch.setattr(
        gen,
        "generation_service",
        SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda t: [7] * len(t))),
    )

    # The backend drains 3 tokens ('a','b','c') through the stop buffer and
    # yields them as one chunk.
    def _stop_buffered_backend(**_kw):
        yield "abc"

    monkeypatch.setattr(gen, "_sync_stream_generate", _stop_buffered_backend)

    async def _collect(agen):
        return [c async for c in agen]

    collected = asyncio.run(
        _collect(gen._stream_generator(GenerationRequest(prompt="p", max_new_tokens=3, stop="ZZZ")))
    )

    assert "".join(collected) == "abc"
    counter_value = m.tokens_generated_total.labels(endpoint="generate")._value.get()
    assert counter_value == 3, f"expected 3 decoded tokens in one chunk, observed {counter_value}"
