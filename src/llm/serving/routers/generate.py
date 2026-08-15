"""Text generation endpoints (``/generate``, ``/batch_generate``)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from llm.serving.auth import get_api_key
from llm.serving.config import ServingConfig
from llm.serving.errors import APIError, ErrorCode
from llm.serving.metrics import METRICS, ServingMetrics
from llm.serving.schemas import (
    BatchGenerationRequest,
    BatchGenerationResponse,
    GenerationRequest,
    GenerationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"])


@contextlib.asynccontextmanager
async def _null_cm():
    """Async no-op context manager for when the semaphore is not configured.

    Used as a fallback in request handlers so the ``async with`` block is
    always valid even if ``configure()`` was never called (which would be
    a programming error caught by ``_require_generation_service`` later).
    """
    yield


# Module-level references are set by ``llm.serving.api`` during lifespan
# startup. Keeping them here (instead of in the router closure) preserves
# the same module-level shape as before, so any future process-level
# instrumentation (e.g. a /debug route) can reach them without
# dependency-injection refactors.
config: ServingConfig | None = None
generation_service = None  # ServingGenerationService; see llm.serving.generation_service
inference_semaphore: asyncio.Semaphore | None = None
# ``metrics`` is bound to the module-level singleton by default so tests
# that import this module without calling ``configure`` still observe
# their writes (against the default registry). Lifespan re-binds to the
# same singleton, so production and tests see one consistent set of
# counters.
metrics: ServingMetrics = METRICS


def configure(
    config_: ServingConfig,
    generation_service_,
    semaphore_: asyncio.Semaphore,
    metrics_: ServingMetrics | None = None,
) -> None:
    """Bind the module-level references.

    Called once during FastAPI lifespan startup. Importing this module
    without configuring first will yield None values and the endpoints
    will refuse to serve (the ``RuntimeError`` below is a programming
    error, not a runtime condition).
    """
    global config, generation_service, inference_semaphore, metrics
    config = config_
    generation_service = generation_service_
    inference_semaphore = semaphore_
    if metrics_ is not None:
        metrics = metrics_


def _require_generation_service():
    if generation_service is None:
        raise RuntimeError("Generation service not initialized")
    return generation_service


# Own thread pool for the streaming bridge instead of ``asyncio.to_thread``:
# ``asyncio.to_thread`` uses the loop's default executor, and
# ``asyncio.run``'s ``shutdown_default_executor`` JOINS it — a stalled
# stream's abandoned ``next()`` would then block the whole test/CLI teardown
# for the duration of the stall. A module-level executor is never joined by
# the loop, and abandoned threads are free to finish (and release the sync
# generator's slot) on their own.
_sync_stream_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm-sync-stream")


def _close_sync_iterator(iterator: Any) -> None:
    """Best-effort ``close()`` of a sync generator (its ``finally`` reap).

    ``close()`` cannot interrupt an executing generator (``ValueError``) and
    is a no-op on an exhausted one (``StopIteration``) — both are fine, the
    slot is released exactly when the generator leaves its yield point.
    """
    with contextlib.suppress(StopIteration, ValueError, RuntimeError):
        iterator.close()


async def _drive_sync_iterator(iterator):
    """Yield from ``iterator`` (a sync generator) off the event loop, and
    CLOSE it the moment the async consumer abandons it.

    Starlette's ``iterate_in_threadpool`` never closes the sync generator
    when the async consumer stops pulling (idle timeout / disconnect), so a
    timeout-aborted stream leaves the engine's ``stream_request`` generator
    suspended mid-generation — holding its KV/scheduler slot and burning
    forward passes until the async generator is finally GC'd (RIL ISS-122,
    F2). This adapter guarantees the sync generator's ``finally`` reap runs
    as soon as the consumer walks away.

    Cancelling the awaiting task does NOT stop the worker thread: the
    in-flight ``next()`` is a bounded engine step. A done-callback on that
    future closes the generator as soon as the step returns (in the worker
    thread itself), so the reap is prompt and does not depend on the event
    loop staying alive. The 504/error response is never blocked waiting for
    the stalled forward.

    Yields each chunk. Raises ``StopAsyncIteration`` at exhaustion and
    propagates the worker exception (or ``CancelledError`` after the close
    has been scheduled).
    """
    from concurrent.futures import Future

    pending: Future | None = None

    def _next() -> tuple[bool, Any]:
        # Convert StopIteration into a sentinel result: raw StopIteration
        # cannot propagate through a Future ("interacts badly with
        # generators").
        try:
            return False, next(iterator)
        except StopIteration:
            return True, None

    def _abandon() -> None:
        # Arranges the close to run exactly when the in-flight next() in the
        # worker thread returns. A raw concurrent.futures done-callback fires
        # from the worker thread itself, so the reap does NOT depend on the
        # event loop staying alive (the route teardown / asyncio.run may have
        # already closed it).
        if pending is None or pending.done():
            _close_sync_iterator(iterator)
        else:
            pending.add_done_callback(lambda _f: _close_sync_iterator(iterator))

    try:
        while True:
            pending = _sync_stream_executor.submit(_next)
            wrapped = asyncio.wrap_future(pending)
            try:
                exhausted, chunk = await asyncio.shield(wrapped)
            except asyncio.CancelledError, GeneratorExit:
                # Consumer gave up (idle timeout / disconnect / route teardown).
                # Reap the generator once the abandoned forward returns; the
                # error response is not held up by it.
                _abandon()
                raise
            if exhausted:
                return
            yield chunk
    finally:
        # Belt-and-suspenders: if the consumer drops this async generator
        # without a cancellation (plain GC of the bridge), close the sync
        # iterator here too.
        _close_sync_iterator(iterator)


def _validate_prompt_encodable(prompt: str) -> None:
    """Raise a 400 ``APIError`` when the tokenizer cannot encode ``prompt``.

    A character-level tokenizer raises ``KeyError`` for any character outside
    its corpus (a client-caused condition). On the streaming path this must be
    validated BEFORE the SSE starts — an error raised inside the generator
    would otherwise surface as a 200 stream containing ``Error: KeyError``
    (RIL ISS-113). Non-streaming handlers catch the same ``KeyError`` in their
    ``except`` clauses.
    """
    service = _require_generation_service()
    try:
        service.tokenizer.encode(prompt)
    except KeyError as exc:
        raise APIError(
            ErrorCode.INVALID_REQUEST,
            f"Invalid request: {exc}",
            details={"field": str(exc)},
        ) from exc


def _sync_generate(prompt: str, **kwargs) -> str:
    service = _require_generation_service()
    return service.generate(prompt=prompt, **kwargs)


def _sync_stream_generate(prompt: str, **kwargs):
    service = _require_generation_service()
    yield from service.stream(prompt=prompt, **kwargs)


def _sync_batch_generate(prompts: list[str], **kwargs) -> list[str]:
    service = _require_generation_service()
    return service.batch_generate(prompts=prompts, **kwargs)


@router.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    config_: Annotated[ServingConfig, Depends(lambda: config)],
    _api_key: Annotated[str, Depends(get_api_key)],
) -> GenerationResponse | StreamingResponse:
    """Generate text from a single prompt. Supports streaming and non-streaming."""
    if request.stream:
        # Validate before the SSE starts so an un-encodable prompt is a real
        # 4xx, not an "Error: ..." chunk inside a 200 stream (RIL ISS-113).
        _validate_prompt_encodable(request.prompt)
        return StreamingResponse(_stream_generator(request), media_type="text/event-stream")

    timer = metrics.request_timer(endpoint="generate")
    with timer as t:
        try:
            async with asyncio.timeout(config_.request_timeout):
                async with inference_semaphore or _null_cm():
                    with metrics.track_inflight():
                        generated_text = await run_in_threadpool(
                            _sync_generate,
                            prompt=request.prompt,
                            max_new_tokens=request.max_new_tokens,
                            temperature=request.temperature,
                            top_k=request.top_k,
                            top_p=request.top_p,
                            repetition_penalty=request.repetition_penalty,
                            frequency_penalty=request.frequency_penalty,
                            presence_penalty=request.presence_penalty,
                            logit_bias=request.logit_bias,
                            stop=request.stop,
                        )
        except TimeoutError as exc:
            t.set_status(504)
            raise APIError(ErrorCode.TIMEOUT, "Request timeout") from exc
        except RuntimeError as exc:
            t.set_status(503)
            raise APIError(ErrorCode.MODEL_UNAVAILABLE, str(exc)) from exc
        except (ValueError, KeyError) as exc:
            t.set_status(400)
            raise APIError(ErrorCode.INVALID_REQUEST, f"Invalid request: {exc}", details={"field": str(exc)}) from exc
        except APIError as exc:
            t.set_status(exc.status_code)
            raise
        except Exception as exc:
            logger.exception("Unexpected error in generate_text")
            t.set_status(500)
            raise APIError(ErrorCode.INTERNAL, "Internal server error") from exc
        else:
            t.set_status(200)
    metrics.observe_tokens(endpoint="generate", token_count=len(generated_text))
    return GenerationResponse(generated_text=generated_text, token_count=len(generated_text))


async def _stream_generator(request: GenerationRequest) -> AsyncGenerator[str]:
    """Stream tokens from the sync engine as an SSE-friendly async iterable.

    An IDLE timeout (``config.request_timeout`` with no chunk produced) is
    applied per chunk: a stream that keeps producing tokens is never cut
    off, but a stalled/misbehaving generator must not hold the inference
    semaphore slot forever — with ``max_concurrent_requests`` stuck streams
    the whole API would stop admitting new requests (RIL sweep finding).
    """
    timer = metrics.request_timer(endpoint="generate")
    token_count = 0
    timed_out = False
    with timer as t:
        try:
            # Acquire the inference semaphore for the *lifetime of the
            # stream*, exactly like the chat streaming route does. The
            # original concurrency-control fix (RIL ISS-036) scoped the
            # semaphore to the non-streaming handlers and chat streaming
            # but missed this route — /generate?stream=true could hold
            # unlimited concurrent generations past
            # max_concurrent_requests while peer routes were bounded
            # (RIL ISS-042).
            async with inference_semaphore or _null_cm():
                with metrics.track_inflight():
                    iterator = _sync_stream_generate(
                        prompt=request.prompt,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        repetition_penalty=request.repetition_penalty,
                        frequency_penalty=request.frequency_penalty,
                        presence_penalty=request.presence_penalty,
                        logit_bias=request.logit_bias,
                        stop=request.stop,
                    )
                    timeout_s = config.request_timeout if config is not None else 60.0
                    async_chunks = _drive_sync_iterator(iterator)
                    while True:
                        try:
                            async with asyncio.timeout(timeout_s):
                                chunk = await async_chunks.__anext__()
                        except StopAsyncIteration:
                            break
                        except TimeoutError:
                            # No token for the whole window — abort the stream
                            # and release the semaphore slot.
                            # ``_drive_sync_iterator`` closes the abandoned
                            # sync generator in the background so
                            # ``stream_request``'s finally-reap (scheduler
                            # remove + slot release) runs promptly instead of
                            # at asyncgen GC (RIL ISS-122).
                            timed_out = True
                            t.set_status(504)
                            yield "Error: stream timed out (no tokens for the request_timeout window)"
                            break
                        token_count += 1
                        yield chunk
            if not timed_out:
                t.set_status(200)
                metrics.observe_tokens(endpoint="generate", token_count=token_count)
        except Exception as exc:
            logger.exception("Error in stream generation")
            t.set_status(500)
            yield f"Error: {type(exc).__name__}"


@router.post("/batch_generate", response_model=BatchGenerationResponse)
async def batch_generate_text(
    request: BatchGenerationRequest,
    config_: Annotated[ServingConfig, Depends(lambda: config)],
    _api_key: Annotated[str, Depends(get_api_key)],
) -> BatchGenerationResponse:
    """Generate text for a batch of prompts in one call."""
    timer = metrics.request_timer(endpoint="batch_generate")
    with timer as t:
        try:
            async with asyncio.timeout(config_.request_timeout):
                async with inference_semaphore or _null_cm():
                    with metrics.track_inflight():
                        results = await run_in_threadpool(
                            _sync_batch_generate,
                            prompts=request.prompts,
                            max_new_tokens=request.max_new_tokens,
                            temperature=request.temperature,
                            top_k=request.top_k,
                            top_p=request.top_p,
                            repetition_penalty=request.repetition_penalty,
                            frequency_penalty=request.frequency_penalty,
                            presence_penalty=request.presence_penalty,
                            logit_bias=request.logit_bias,
                            stop=request.stop,
                        )
        except TimeoutError as exc:
            t.set_status(504)
            raise APIError(ErrorCode.TIMEOUT, "Request timeout") from exc
        except RuntimeError as exc:
            t.set_status(503)
            raise APIError(ErrorCode.MODEL_UNAVAILABLE, str(exc)) from exc
        except (ValueError, KeyError) as exc:
            t.set_status(400)
            raise APIError(ErrorCode.INVALID_REQUEST, f"Invalid request: {exc}", details={"field": str(exc)}) from exc
        except APIError as exc:
            t.set_status(exc.status_code)
            raise
        except Exception as exc:
            logger.exception("Unexpected error in batch_generate_text")
            t.set_status(500)
            raise APIError(ErrorCode.INTERNAL, "Internal server error") from exc
        else:
            t.set_status(200)
    # Record per-prompt token count; the counter is cumulative across
    # the whole batch, the histogram is per-prompt.
    for text in results:
        metrics.observe_tokens(endpoint="batch_generate", token_count=len(text))
    return BatchGenerationResponse(
        results=[GenerationResponse(generated_text=text, token_count=len(text)) for text in results]
    )
