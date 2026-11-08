"""Tests for :mod:`llm.serving.metrics` (T2 #22, Finding AX).

Covers the structure of the ``ServingMetrics`` collection (counters,
histograms, gauges, labels, buckets) and the convenience helpers used
by the serving tier to record observations. Uses a private
``CollectorRegistry`` per test so we never collide with the default
Prometheus registry used by ``/metrics`` in production.
"""

from __future__ import annotations

import pytest
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


# --- Non-finite ratios must be rejected, not exposed ---


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_kv_cache_hit_ratio_rejects_non_finite(bad):
    """A non-finite ratio must raise, not reach the Gauge.

    ``nan < 0.0`` and ``nan > 1.0`` are BOTH False, so a plain range check lets
    NaN through. Prometheus then scrapes the literal string ``NaN``, and one NaN
    is enough to poison every ``avg()`` / ``rate()`` aggregation on a dashboard
    until the next successful step — with no error raised at the recording site.
    ``inf`` slips past the same way for the ``-inf`` side.
    """
    metrics = ServingMetrics(registry=CollectorRegistry())

    with pytest.raises(ValueError, match="finite"):
        metrics.record_kv_cache_hit_ratio(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_batch_fill_ratio_rejects_non_finite(bad):
    """``scheduled / total`` can only be non-finite via a non-finite input."""
    metrics = ServingMetrics(registry=CollectorRegistry())

    with pytest.raises(ValueError, match=r"outside \[0, 1\]|finite"):
        metrics.record_batch_fill_ratio(scheduled=bad, total_active_slots=8)


def test_valid_ratios_still_recorded_after_the_guard():
    """The guard must not reject the values it is meant to allow."""
    from prometheus_client import generate_latest

    registry = CollectorRegistry()
    metrics = ServingMetrics(registry=registry)

    metrics.record_kv_cache_hit_ratio(0.0)
    metrics.record_kv_cache_hit_ratio(0.5)
    metrics.record_kv_cache_hit_ratio(1.0)
    metrics.record_batch_fill_ratio(scheduled=0, total_active_slots=8)
    metrics.record_batch_fill_ratio(scheduled=8, total_active_slots=8)

    exposed = generate_latest(registry).decode()
    assert "NaN" not in exposed, "a NaN reached /metrics"
    assert "Inf" not in exposed, "an Inf reached /metrics"


def test_request_timer_defaults_to_the_error_status_label():
    """A timer whose ``set_status`` was never called must still be labelled.

    ``_RequestTimer`` documents "Defaults to ``status=\\"error\\"`` so a forgotten
    call still emits a labelled observation". Without that default an exception
    path that forgets to set a status would either crash on exit or record an
    unlabelled observation, and the request would vanish from the per-status
    dashboards instead of showing up as an error.
    """
    import time

    m = _fresh()
    with m.request_timer(endpoint="generate"):
        time.sleep(0.001)
        # Deliberately do NOT call set_status.

    count, total = _histogram_count_and_sum(m.request_duration_seconds, endpoint="generate", status="error")
    assert count == 1, "an unlabelled exit must still record under status='error'"
    assert total >= 0.0


def test_request_timer_records_on_the_exception_path():
    """An exception inside the block must still produce an observation.

    The histogram is written in ``__exit__``, so the request is counted whether
    or not the handler completed — that is what makes latency dashboards able to
    distinguish fast failures from hangs.
    """
    import pytest

    m = _fresh()

    def _failing_request() -> None:
        with m.request_timer(endpoint="chat") as timer:
            timer.set_status(500)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _failing_request()

    count, _ = _histogram_count_and_sum(m.request_duration_seconds, endpoint="chat", status="500")
    assert count == 1, "the exception path must still record timing"


def test_batch_fill_ratio_tracks_the_engine_denominator():
    """The ratio must be scheduled / configured-slot-count, not scheduled / running.

    ``StepStats.total_active_slots`` is the engine's full slot pool, so the gauge
    reads 0.25 for 2-of-8 even when only 2 sequences are alive. Using the live
    count as the denominator would always read 1.0 and hide under-utilisation —
    the exact signal the metric exists to expose.
    """
    m = _fresh()
    for scheduled, total, expected in [(0, 8, 0.0), (2, 8, 0.25), (8, 8, 1.0)]:
        m.record_batch_fill_ratio(scheduled=scheduled, total_active_slots=total)
        value = m.batch_fill_ratio._value.get()
        assert value == expected, f"{scheduled}/{total} recorded {value}, expected {expected}"
