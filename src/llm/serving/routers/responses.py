"""OpenAI-compatible Responses API endpoint (``POST /v1/responses``).

OpenAI recommends the Responses API over chat completions for new integrations,
so a compatibility layer that only speaks ``/v1/chat/completions`` leaves the
current client generation without a target. This route serves the text-only
subset: the SAME generation path as the chat endpoint, presented in the
Responses API's shape.

What the shape changes (and why this is not a thin alias):

* the prompt arrives as ``input`` (a string or a message list) plus a separate
  ``instructions`` string, instead of a ``messages`` array where instructions
  ride as a ``system`` role;
* the budget is ``max_output_tokens``;
* the text comes back inside ``output``, a list of items each holding a
  ``content`` list, rather than ``choices[0].message.content``;
* truncation is reported as ``status="incomplete"`` with
  ``incomplete_details.reason="max_output_tokens"``, not ``finish_reason``.

Streaming uses the event vocabulary the SDK's ``client.responses.stream()``
consumes: ``response.created``, ``response.output_text.delta`` (repeated),
``response.completed`` / ``response.incomplete``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from llm.serving.auth import get_api_key
from llm.serving.batch_engine import NoFreeSlotsError
from llm.serving.chat_template import messages_to_prompt
from llm.serving.config import ServingConfig
from llm.serving.errors import DEFAULT_QUEUE_FULL_RETRY_AFTER_SECONDS, APIError, ErrorCode
from llm.serving.metrics import METRICS, ServingMetrics
from llm.serving.routers.generate import (
    _drive_sync_iterator,
    _sync_generate,
    _sync_stream_generate,
    _token_count,
    _validate_generation_bounds,
    _validate_queue_has_capacity,
    _validate_stream_request,
)
from llm.serving.scheduler import QueueFullError
from llm.serving.schemas import (
    ChatMessage as _ChatMessage,
)
from llm.serving.schemas import (
    ResponsesIncompleteDetails,
    ResponsesOutputMessage,
    ResponsesOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["responses"])

# Same defaults the chat endpoint documents: 50 tokens unless the client asks
# for something else. Kept in one place so the two endpoints cannot drift.
DEFAULT_MAX_OUTPUT_TOKENS = 50


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


config: ServingConfig | None = None
inference_semaphore: asyncio.Semaphore | None = None
metrics: ServingMetrics = METRICS


@contextlib.asynccontextmanager
async def _null_cm():
    """Stand-in for the semaphore when the module is used without lifespan startup."""
    yield


def _to_chat_messages(request: ResponsesRequest) -> list[_ChatMessage]:
    """Lower a Responses request onto the chat-message list this server renders.

    ``instructions`` becomes a leading system message — that is the mapping
    OpenAI documents, and it keeps ONE prompt renderer for both endpoints
    (``ServingConfig.chat_message_template`` and the generation prefix therefore
    apply identically, which is what makes the two surfaces comparable).

    ``developer`` is the Responses-era spelling of ``system``; it is folded into
    ``system`` rather than dropped, because dropping it would silently discard
    the client's highest-priority text.
    """
    messages: list[_ChatMessage] = []
    if request.instructions:
        messages.append(_ChatMessage(role="system", content=request.instructions))

    if isinstance(request.input, str):
        messages.append(_ChatMessage(role="user", content=request.input))
        return messages

    for item in request.input:
        role = "system" if item.role in ("system", "developer") else item.role
        messages.append(_ChatMessage(role=role, content=item.content))
    return messages


def _build_response(
    *,
    request: ResponsesRequest,
    completion: str,
    completion_tokens: int,
    prompt_tokens: int,
    max_output_tokens: int,
) -> ResponsesResponse:
    """Assemble the Responses-API object from the generated text."""
    # The budget was fully consumed -> the SDK's clients check ``status`` and
    # ``incomplete_details``; ``finish_reason`` has no counterpart here.
    truncated = completion_tokens >= max_output_tokens
    status = "incomplete" if truncated else "completed"
    output = [
        ResponsesOutputMessage(
            status=status,
            content=[ResponsesOutputText(text=completion)],
        )
    ]
    return ResponsesResponse(
        model=request.model,
        status=status,
        output=output,
        output_text=completion,
        usage=ResponsesUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        incomplete_details=ResponsesIncompleteDetails(reason="max_output_tokens") if truncated else None,
        temperature=request.temperature,
        top_p=request.top_p,
        max_output_tokens=max_output_tokens,
        instructions=request.instructions,
    )


@router.post("/v1/responses", response_model=ResponsesResponse)
async def create_response(
    request: ResponsesRequest,
    config_: Annotated[ServingConfig, Depends(lambda: config)],
    _api_key: Annotated[str, Depends(get_api_key)],
) -> ResponsesResponse | StreamingResponse:
    """OpenAI-compatible Responses API endpoint."""
    messages = _to_chat_messages(request)
    prompt = messages_to_prompt(
        messages,
        message_template=config_.chat_message_template,
        generation_prefix=config_.chat_generation_prefix,
    )
    max_output_tokens = request.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS
    temperature = 1.0 if request.temperature is None else request.temperature

    if request.stream:
        _validate_stream_request(prompt, max_output_tokens, budget_field="max_output_tokens")
        return StreamingResponse(
            _responses_stream_generator(
                request=request,
                prompt=prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            ),
            media_type="text/event-stream",
        )

    _validate_generation_bounds(prompt, max_output_tokens, budget_field="max_output_tokens")
    _validate_queue_has_capacity()

    timer = metrics.request_timer(endpoint="responses")
    with timer as t:
        try:
            async with asyncio.timeout(config_.request_timeout):
                async with inference_semaphore or _null_cm():
                    with metrics.track_inflight():
                        generated_text = await run_in_threadpool(
                            _sync_generate,
                            prompt=prompt,
                            max_new_tokens=max_output_tokens,
                            temperature=temperature,
                            top_p=request.top_p,
                        )
        except TimeoutError as exc:
            t.set_status(504)
            raise APIError(ErrorCode.TIMEOUT, "Request timeout") from exc
        except (QueueFullError, NoFreeSlotsError) as exc:
            # Both are backpressure, not model failure: retryable 503s. They
            # subclass RuntimeError, so they must be caught before it.
            t.set_status(503)
            raise APIError(
                ErrorCode.QUEUE_FULL,
                "server is at capacity; retry later",
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
            logger.exception("Unexpected error in responses")
            t.set_status(500)
            raise APIError(ErrorCode.INTERNAL, "Internal server error") from exc
        else:
            t.set_status(200)

    completion = generated_text[len(prompt) :] if generated_text.startswith(prompt) else generated_text
    completion_tokens = _token_count(completion)
    metrics.observe_tokens(endpoint="responses", token_count=completion_tokens)

    return _build_response(
        request=request,
        completion=completion,
        completion_tokens=completion_tokens,
        prompt_tokens=_token_count(prompt),
        max_output_tokens=max_output_tokens,
    )


def _sse(event: dict) -> str:
    """Frame one Responses API stream event as SSE."""
    import json

    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


async def _responses_stream_generator(
    *,
    request: ResponsesRequest,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> AsyncGenerator[str]:
    """Stream the Responses API event sequence.

    The SDK's ``client.responses.stream()`` accumulates the deltas into a
    snapshotted ``Response``, and that accumulator fixes the order: a
    ``response.output_text.delta`` is applied to ``output[output_index]
    .content[content_index]``, so the corresponding
    ``response.content_part.added`` MUST arrive first (measured: skipping it
    raises ``IndexError: list index out of range`` inside the SDK). The full
    sequence emitted here is

        response.created
        response.output_item.added
        response.content_part.added
        response.output_text.delta   x N
        response.output_text.done
        response.content_part.done
        response.output_item.done
        response.completed | response.incomplete
    """
    response_id = f"resp_{uuid.uuid4().hex[:16]}"
    created_at = int(time.time())
    item_id = f"msg_{uuid.uuid4().hex[:12]}"
    prompt_tokens = _token_count(prompt)
    sequence = 0

    def snapshots(status: str, output: list[dict], text: str, completion_tokens: int) -> dict:
        """The ``response`` object embedded in each event."""
        truncated = completion_tokens >= max_output_tokens
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": request.model,
            "status": status,
            "output": output,
            "output_text": text,
            "usage": {
                "input_tokens": prompt_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": completion_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_output_tokens": max_output_tokens,
            "instructions": request.instructions,
            "error": None,
        }

    def text_part(text: str) -> dict:
        return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}

    yield _sse(
        {
            "type": "response.created",
            "sequence_number": sequence,
            "response": snapshots("in_progress", [], "", 0),
        }
    )
    sequence += 1

    yield _sse(
        {
            "type": "response.output_item.added",
            "sequence_number": sequence,
            "output_index": 0,
            "item": {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []},
        }
    )
    sequence += 1

    yield _sse(
        {
            "type": "response.content_part.added",
            "sequence_number": sequence,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": text_part(""),
        }
    )
    sequence += 1

    text = ""
    token_count = 0
    iterator = None
    failed = False
    try:
        iterator = _sync_stream_generate(prompt=prompt, max_new_tokens=max_output_tokens, temperature=temperature)
        # ``_drive_sync_iterator`` is an ASYNC GENERATOR wrapping the sync one (it
        # closes the sync generator when the consumer abandons it, releasing the
        # engine's KV slot). Awaiting it raises "'async_generator' object can't be
        # awaited" -- it must be consumed with ``async for``.
        async for token in _drive_sync_iterator(iterator):
            text += token
            token_count += _token_count(token)
            yield _sse(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": sequence,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": token,
                }
            )
            sequence += 1
    except TimeoutError:
        failed = True
    finally:
        if iterator is not None:
            with contextlib.suppress(Exception):
                iterator.close()

    if failed:
        yield _sse(
            {
                "type": "response.failed",
                "sequence_number": sequence,
                "response": snapshots("failed", [], text, token_count),
            }
        )
        return

    truncated = token_count >= max_output_tokens
    final_status = "incomplete" if truncated else "completed"
    item = {
        "id": item_id,
        "type": "message",
        "status": final_status,
        "role": "assistant",
        "content": [text_part(text)],
    }

    yield _sse(
        {
            "type": "response.output_text.done",
            "sequence_number": sequence,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": text,
            "logprobs": [],
        }
    )
    sequence += 1

    yield _sse(
        {
            "type": "response.content_part.done",
            "sequence_number": sequence,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": text_part(text),
        }
    )
    sequence += 1

    yield _sse(
        {
            "type": "response.output_item.done",
            "sequence_number": sequence,
            "output_index": 0,
            "item": item,
        }
    )
    sequence += 1

    metrics.observe_tokens(endpoint="responses", token_count=token_count)
    yield _sse(
        {
            "type": "response.incomplete" if truncated else "response.completed",
            "sequence_number": sequence,
            "response": snapshots(final_status, [item], text, token_count),
        }
    )
