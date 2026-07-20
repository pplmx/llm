"""Custom Prometheus metrics for the serving API (Finding AX, T2 #22).

``prometheus-fastapi-instrumentator`` already gives generic HTTP RED
metrics (rate, errors, duration) per route. Domain-specific signals —
tokens generated per request, batch fill ratio, KV-cache hit rate,
queue depth — are not visible to operators without these counters.

Layout
------

:class:`ServingMetrics` is a thin container around a ``CollectorRegistry``
that holds the six metrics required by the audit ticket. The module
also exposes a module-level :data:`METRICS` singleton wired into the
default Prometheus registry so :func:`prometheus_fastapi_instrumentator.Instrumentator.expose`
makes them visible at ``/metrics``.

Usage
-----

::

    from llm.serving.metrics import METRICS

    with METRICS.request_timer(endpoint="generate") as timer:
        async with METRICS.track_inflight():
            text = await run_in_threadpool(_sync_generate, ...)
        METRICS.observe_tokens(endpoint="generate", token_count=len(text))
        timer.set_status(200)

The engine can publish per-step stats via :func:`ServingMetrics.record_batch_fill_ratio`,
called from the ``on_step`` hook passed to
:class:`~llm.serving.batch_engine.ContinuousBatchingEngine`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Protocol

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

# Histogram bucket upper bounds, taken verbatim from the audit ticket.
# ``llm_tokens_per_request`` spans interactive (16) up to long-form (4096).
_TOKEN_BUCKETS = (16.0, 64.0, 256.0, 1024.0, 4096.0)
# ``llm_request_duration_seconds`` spans sub-50ms p50 up to 30s tail.
_DURATION_BUCKETS = (0.05, 0.25, 1.0, 5.0, 30.0)


class _StatusTimer(Protocol):
    """Minimal timer protocol returned by :meth:`ServingMetrics.request_timer`."""

    def set_status(self, status: int) -> None: ...


class _RequestTimer:
    """Internal helper returned by :meth:`ServingMetrics.request_timer`.

    Records the wall-clock duration when the context exits, paired with
    the HTTP ``status`` label set via :meth:`set_status` before exit.
    Defaults to ``status="error"`` so a forgotten call still emits a
    labelled observation.
    """

    __slots__ = ("_endpoint", "_histogram", "_start", "_status")

    def __init__(self, histogram: Histogram, endpoint: str) -> None:
        self._histogram = histogram
        self._endpoint = endpoint
        self._start = 0.0
        self._status = "error"

    def set_status(self, status: int) -> None:
        self._status = str(status)

    def _observe(self, end: float) -> None:
        self._histogram.labels(endpoint=self._endpoint, status=self._status).observe(end - self._start)

    def __enter__(self) -> _RequestTimer:
        import time

        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        import time

        self._observe(time.perf_counter())


class ServingMetrics:
    """Container for the serving tier's domain Prometheus metrics.

    Args:
        registry: The Prometheus registry to register metrics against.
            Defaults to the module-level default (:data:`prometheus_client.REGISTRY`)
            so ``/metrics`` exposes them. Tests should pass a private
            :class:`CollectorRegistry` for isolation.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry: CollectorRegistry = registry if registry is not None else REGISTRY

        self.tokens_generated_total: Counter = Counter(
            "llm_tokens_generated_total",
            "Total completion tokens generated, by endpoint.",
            labelnames=("endpoint",),
            registry=self.registry,
        )
        self.tokens_per_request: Histogram = Histogram(
            "llm_tokens_per_request",
            "Distribution of completion tokens per request, by endpoint.",
            labelnames=("endpoint",),
            buckets=_TOKEN_BUCKETS,
            registry=self.registry,
        )
        self.request_duration_seconds: Histogram = Histogram(
            "llm_request_duration_seconds",
            "End-to-end request duration, by endpoint and HTTP status.",
            labelnames=("endpoint", "status"),
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.batch_fill_ratio: Gauge = Gauge(
            "llm_batch_fill_ratio",
            "Fraction of continuous-batching slots currently in use (0-1).",
            registry=self.registry,
        )
        self.kv_cache_hit_ratio: Gauge = Gauge(
            "llm_kv_cache_hit_ratio",
            "Fraction of recent prefix lookups served from the KV prefix cache (0-1).",
            registry=self.registry,
        )
        self.inflight_requests: Gauge = Gauge(
            "llm_inflight_requests",
            "Number of in-flight inference requests currently holding a slot.",
            registry=self.registry,
        )

    # --- Observation helpers ---------------------------------------------

    def observe_tokens(self, *, endpoint: str, token_count: int) -> None:
        """Record one request's completion token count.

        Bumps both the cumulative counter and the per-request histogram
        so dashboards can show throughput and p95 in one set of queries.
        """
        if token_count < 0:
            raise ValueError(f"token_count must be non-negative, got {token_count}")
        self.tokens_generated_total.labels(endpoint=endpoint).inc(token_count)
        self.tokens_per_request.labels(endpoint=endpoint).observe(token_count)

    def record_batch_fill_ratio(self, *, scheduled: int, total_active_slots: int) -> None:
        """Update ``llm_batch_fill_ratio`` from a ``step()`` result.

        ``total_active_slots`` is the engine's full slot pool; ``scheduled``
        is how many of those were used in the most recent step. Callers
        passing ``total_active_slots=0`` get a ``ValueError`` because the
        ratio is undefined (engine not yet initialized).
        """
        if total_active_slots <= 0:
            raise ValueError(f"Cannot compute batch fill ratio: total_active_slots={total_active_slots}")
        ratio = scheduled / total_active_slots
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(
                f"Batch fill ratio {ratio} is outside [0, 1] "
                f"(scheduled={scheduled}, total_active_slots={total_active_slots})"
            )
        self.batch_fill_ratio.set(ratio)

    def record_kv_cache_hit_ratio(self, ratio: float) -> None:
        """Update ``llm_kv_cache_hit_ratio``. ``ratio`` must be in [0, 1]."""
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"KV cache hit ratio {ratio} is outside [0, 1]")
        self.kv_cache_hit_ratio.set(ratio)

    @contextmanager
    def track_inflight(self) -> Iterator[None]:
        """Context manager that inc/dec ``llm_inflight_requests``.

        Use to wrap the section of a request that holds an inference
        slot (typically ``async with inference_semaphore``). Decrement
        runs even on exception so the gauge never sticks above zero.
        """
        self.inflight_requests.inc()
        try:
            yield
        finally:
            self.inflight_requests.dec()

    def request_timer(self, *, endpoint: str) -> _RequestTimer:
        """Return a context manager that times a request and labels by status.

        Use :meth:`_RequestTimer.set_status` before exiting to record the
        HTTP status; the default is ``"error"`` so a missing call still
        emits a labelled observation.
        """
        return _RequestTimer(self.request_duration_seconds, endpoint)


# Module-level singleton wired into the default registry. The serving
# tier (``api.py``, routers) imports this directly; tests build their
# own :class:`ServingMetrics` with a private registry for isolation.
METRICS = ServingMetrics()

__all__ = ["METRICS", "ServingMetrics"]
