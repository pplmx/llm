"""OpenAI-compatible legacy completions endpoint (``POST /v1/completions``).

The chat surface (``/v1/chat/completions``) is what most clients use today,
but the legacy ``/v1/completions`` route is still spoken by a real share of
the ecosystem: the official SDK's ``client.completions.create``, LangChain's
non-chat ``OpenAI`` wrapper, prompt-template pipelines, and a long tail of
evaluation harnesses. Pointed at this server those clients got a bare 404
even though generation worked, so the route is exposed here as a thin
adapter over the shared
:class:`~llm.serving.generation_service.ServingGenerationService`.

Contract notes (all matching OpenAI's documented shape):

* ``prompt`` is either a string or a list of strings. A list yields one
  choice **per prompt, in order** — not one concatenated request.
* ``choices[i].text`` is the **completion only**; the prompt is never echoed,
  which is the opposite of this project's own ``/generate`` route (that one
  strips the echo for the same reason). The eager backend's ``generate()``
  returns ``prompt + completion``, so the prefix is stripped here.
* ``finish_reason`` is ``"length"`` when the ``max_tokens`` budget was
  exhausted and ``"stop"`` otherwise, mirroring the chat route.
* ``usage.prompt_tokens`` / ``completion_tokens`` are real tokenizer counts
  (via :func:`llm.serving.routers.generate._token_count`), not character
  counts.

The same pre-stream validations as ``/generate`` run first, so an
un-encodable prompt or an over-window ``max_tokens`` is a real 4xx instead of
an ``Error: ...`` chunk inside a 200 SSE stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from llm.serving.auth import get_api_key
from llm.serving.config import ServingConfig
from llm.serving.errors import DEFAULT_QUEUE_FULL_RETRY_AFTER_SECONDS, APIError, ErrorCode
from llm.serving.metrics import METRICS, ServingMetrics
from llm.serving.routers.generate import (
    _drive_sync_iterator,
    _null_cm,
    _sync_batch_generate,
    _sync_generate,
    _sync_stream_generate,
    _token_count,
    _validate_generation_bounds,
    _validate_prompt_encodable,
    _validate_queue_has_capacity,
    _validate_stream_request,
)
from llm.serving.scheduler import QueueFullError
from llm.serving.schemas import (
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["completions"])

# Bound during lifespan startup, exactly like the chat router: the endpoint
# needs the serving config (timeouts) and the shared inference semaphore so it
# obeys ``max_concurrent_requests`` instead of becoming an unbounded bypass.
config: ServingConfig | None = None
inference_semaphore: asyncio.Semaphore | None = None
metrics: ServingMetrics = METRICS


def configure(
    config_: ServingConfig,
    semaphore_: asyncio.Semaphore,
    metrics_: ServingMetrics | None = None,
) -> None:
    """Bind the module-level config reference, semaphore, and metrics."""
    global config, inference_semaphore, metrics
    config = config_
    inference_semaphore = semaphore_
    if metrics_ is not None:
        metrics = metrics_


def _as_prompts(prompt: str | list[str]) -> list[str]:
    """Normalize the request's ``prompt`` field to a list of prompt strings."""
    return [prompt] if isinstance(prompt, str) else list(prompt)


def _strip_prompt_echo(prompt: str, generated_text: str) -> str:
    """Drop the echoed prompt from a backend's ``generate()`` result.

    The eager backend's ``generate()`` returns ``prompt + completion``. The
    same ``startswith`` guard the other routes use keeps a model that
    (legitimately) repeats the prompt from being over-trimmed.
    """
    return generated_text[len(prompt) :] if generated_text.startswith(prompt) else generated_text


def _generate_all(prompts: list[str], request: CompletionRequest) -> list[str]:
    """Generate one completion per prompt (runs in the threadpool).

    A single prompt uses ``generate`` — the path every other route uses, so
    backend-specific behaviour (KV cache, batching) is identical. Two or more
    prompts go through ``batch_generate`` so the backend can batch them.
    """
    kwargs: dict[str, Any] = {
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
        "logit_bias": request.logit_bias,
        "stop": request.stop,
    }
    if len(prompts) == 1:
        generated = [_sync_generate(prompt=prompts[0], **kwargs)]
    else:
        generated = list(_sync_batch_generate(prompts=prompts, **kwargs))
    return [_strip_prompt_echo(prompt, text) for prompt, text in zip(prompts, generated, strict=True)]


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    config_: Annotated[ServingConfig, Depends(lambda: config)],
    _api_key: Annotated[str, Depends(get_api_key)],
) -> CompletionResponse | StreamingResponse:
    """Legacy OpenAI completions endpoint (streaming and non-streaming)."""
    prompts = _as_prompts(request.prompt)

    if request.stream:
        # A list prompt is refused rather than streamed: one honest SSE stream
        # carries a single ``index``, and this server generates the prompts one
        # at a time, so interleaving them would emit choice indices the client
        # cannot rely on.
        if len(prompts) > 1:
            raise APIError(
                ErrorCode.INVALID_REQUEST,
                "streaming a list of prompts is not supported; send one prompt per streaming request",
            )
        _validate_stream_request(prompts[0], request.max_tokens)
        return StreamingResponse(
            _completion_stream_generator(request, prompts[0]),
            media_type="text/event-stream",
        )

    # Non-streaming pre-checks mirror the streaming ones so both modes reject
    # the same payload the same way (see ``/generate``).
    for prompt in prompts:
        _validate_prompt_encodable(prompt)
        _validate_generation_bounds(prompt, request.max_tokens)
    _validate_queue_has_capacity()

    timer = metrics.request_timer(endpoint="completions")
    with timer as t:
        try:
            async with asyncio.timeout(config_.request_timeout):
                async with inference_semaphore or _null_cm():
                    with metrics.track_inflight():
                        generated = await run_in_threadpool(_generate_all, prompts, request)
        except TimeoutError as exc:
            t.set_status(504)
            raise APIError(ErrorCode.TIMEOUT, "Request timeout") from exc
        except QueueFullError as exc:
            # MUST precede the generic ``RuntimeError`` catch (QueueFullError
            # subclasses it): backpressure is retryable, not a model failure.
            t.set_status(503)
            raise APIError(
                ErrorCode.QUEUE_FULL,
                "server waiting queue is full; retry later",
                retry_after=DEFAULT_QUEUE_FULL_RETRY_AFTER_SECONDS,
                should_retry=True,
            ) from exc
        except RuntimeError as exc:
            t.set_status(503)
            logger.error("backend generation failed: %s", exc)
            raise APIError(ErrorCode.MODEL_UNAVAILABLE, "Model unavailable during generation") from exc
        except (ValueError, KeyError) as exc:
            t.set_status(400)
            raise APIError(ErrorCode.INVALID_REQUEST, f"Invalid request: {exc}", details={"field": str(exc)}) from exc
        except APIError as exc:
            t.set_status(exc.status_code)
            raise
        except Exception as exc:
            logger.exception("Unexpected error in create_completion")
            t.set_status(500)
            raise APIError(ErrorCode.INTERNAL, "Internal server error") from exc
        else:
            t.set_status(200)

    choices = [
        CompletionChoice(
            index=index,
            text=completion,
            finish_reason="length" if _token_count(completion) >= request.max_tokens else "stop",
        )
        for index, completion in enumerate(generated)
    ]
    completion_tokens = sum(_token_count(choice.text) for choice in choices)
    prompt_tokens = sum(_token_count(prompt) for prompt in prompts)
    metrics.observe_tokens(endpoint="completions", token_count=completion_tokens)
    return CompletionResponse(
        model=request.model,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _completion_stream_generator(
    request: CompletionRequest,
    prompt: str,
) -> AsyncGenerator[str]:
    """SSE stream for one legacy completion.

    Emits OpenAI-shaped ``data:`` frames plus the same ``[DONE]`` sentinel as
    the chat route. An IDLE timeout is applied per chunk so a stalled backend
    cannot hold the inference semaphore slot forever.
    """
    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    timer = metrics.request_timer(endpoint="completions")
    token_count = 0
    with timer as t:
        try:
            timeout_s = config.request_timeout if config is not None else 60.0
            async with inference_semaphore or _null_cm():
                with metrics.track_inflight():
                    iterator = _sync_stream_generate(
                        prompt=prompt,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        frequency_penalty=request.frequency_penalty,
                        presence_penalty=request.presence_penalty,
                        logit_bias=request.logit_bias,
                        stop=request.stop,
                    )
                    async_chunks = _drive_sync_iterator(iterator)
                    while True:
                        try:
                            async with asyncio.timeout(timeout_s):
                                token = await async_chunks.__anext__()
                        except StopAsyncIteration:
                            break
                        except TimeoutError:
                            # No token for the whole window — abort with an
                            # in-band error frame and release the slot. The
                            # bridge closes the abandoned sync generator so the
                            # engine reaps its KV/scheduler slot promptly.
                            t.set_status(504)
                            yield _sse_frame(
                                completion_id,
                                created,
                                request.model,
                                CompletionChunkChoice(
                                    text="Error: stream timed out (no tokens for the request_timeout window)"
                                ),
                            )
                            yield "data: [DONE]\n\n"
                            metrics.observe_tokens(endpoint="completions", token_count=token_count)
                            return
                        # Count decoded tokens, not chunks: with ``stop``
                        # sequences the backend flushes several tokens through
                        # its stop buffer as one multi-token chunk.
                        token_count += _token_count(token)
                        yield _sse_frame(
                            completion_id,
                            created,
                            request.model,
                            CompletionChunkChoice(text=token),
                        )

            finish_reason = "length" if token_count >= request.max_tokens else "stop"
            yield _sse_frame(
                completion_id,
                created,
                request.model,
                CompletionChunkChoice(finish_reason=finish_reason),
            )
            yield "data: [DONE]\n\n"
            t.set_status(200)
            metrics.observe_tokens(endpoint="completions", token_count=token_count)
        except Exception:
            logger.exception("Error in completions stream generation")
            t.set_status(500)
            metrics.observe_tokens(endpoint="completions", token_count=token_count)
            # Do NOT echo the exception class name back to the client.
            yield _sse_frame(
                completion_id,
                created,
                request.model,
                CompletionChunkChoice(text="Error: stream failed"),
            )
            yield "data: [DONE]\n\n"


def _sse_frame(
    completion_id: str,
    created: int,
    model: str,
    choice: CompletionChunkChoice,
) -> str:
    """Render one ``data:`` SSE frame for the completions stream."""
    chunk = CompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[choice],
    )
    return f"data: {json.dumps(chunk.model_dump(mode='json'))}\n\n"
