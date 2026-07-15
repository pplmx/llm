"""OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from llm.serving.auth import get_api_key
from llm.serving.chat_template import messages_to_prompt
from llm.serving.config import ServingConfig
from llm.serving.errors import APIError, ErrorCode
from llm.serving.metrics import METRICS, ServingMetrics
from llm.serving.routers.generate import (
    _sync_generate,
    _sync_stream_generate,
)
from llm.serving.schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

config: ServingConfig | None = None
# Bound during lifespan startup so the chat endpoint shares the same
# concurrency cap as /generate. The chat router's configure() takes the
# semaphore as a parameter (instead of importing the generate router's
# copy at import time) so the None→Semaphore transition is atomic from
# the chat router's point of view.
inference_semaphore: asyncio.Semaphore | None = None
# Defaults to the module-level singleton so tests that import the
# router without calling ``configure`` still record into the same
# registry that ``/metrics`` exposes.
metrics: ServingMetrics = METRICS


def configure(
    config_: ServingConfig,
    semaphore_: asyncio.Semaphore,
    metrics_: ServingMetrics | None = None,
) -> None:
    """Bind the module-level config reference, semaphore, and metrics.

    Called during lifespan startup.
    """
    global config, inference_semaphore, metrics
    config = config_
    inference_semaphore = semaphore_
    if metrics_ is not None:
        metrics = metrics_


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    config_: Annotated[ServingConfig, Depends(lambda: config)],
    _api_key: Annotated[str, Depends(get_api_key)],
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    prompt = messages_to_prompt(
        request.messages,
        message_template=config_.chat_message_template,
        generation_prefix=config_.chat_generation_prefix,
    )
    repetition_penalty = 1.0 + request.presence_penalty

    if request.stream:
        return StreamingResponse(
            _chat_stream_generator(request, prompt, repetition_penalty, request.frequency_penalty),
            media_type="text/event-stream",
        )

    timer = metrics.request_timer(endpoint="chat_completions")
    with timer as t:
        try:
            async with asyncio.timeout(config_.request_timeout):
                with metrics.track_inflight():
                    generated_text = await run_in_threadpool(
                        _sync_generate,
                        prompt=prompt,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        repetition_penalty=repetition_penalty,
                        frequency_penalty=request.frequency_penalty,
                    )
        except TimeoutError as exc:
            t.set_status(504)
            raise APIError(ErrorCode.TIMEOUT, "Request timeout") from exc
        except RuntimeError as exc:
            t.set_status(503)
            raise APIError(ErrorCode.MODEL_UNAVAILABLE, str(exc)) from exc
        except ValueError as exc:
            t.set_status(400)
            raise APIError(ErrorCode.INVALID_REQUEST, f"Invalid request: {exc}", details={"field": str(exc)}) from exc
        except APIError as exc:
            t.set_status(exc.status_code)
            raise
        except Exception as exc:
            logger.exception("Unexpected error in chat_completions")
            t.set_status(500)
            raise APIError(ErrorCode.INTERNAL, "Internal server error") from exc
        else:
            t.set_status(200)

    # Strip the prompt prefix if the model echoed it back.
    completion = generated_text[len(prompt) :].strip() if generated_text.startswith(prompt) else generated_text.strip()

    metrics.observe_tokens(endpoint="chat_completions", token_count=len(completion))

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionChoiceMessage(content=completion),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=len(prompt),
            completion_tokens=len(completion),
            total_tokens=len(prompt) + len(completion),
        ),
    )


async def _chat_stream_generator(
    request: ChatCompletionRequest,
    prompt: str,
    repetition_penalty: float,
    frequency_penalty: float,
) -> AsyncGenerator[str]:
    """Generate SSE stream for chat completions."""
    from starlette.concurrency import iterate_in_threadpool

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    timer = metrics.request_timer(endpoint="chat_completions")
    token_count = 0
    with timer as t:
        try:
            # First chunk with role.
            first_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(role="assistant"))],
            )
            yield f"data: {first_chunk.model_dump_json()}\n\n"

            with metrics.track_inflight():
                iterator = _sync_stream_generate(
                    prompt=prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                )

                prompt_sent = False
                async for token in iterate_in_threadpool(iterator):
                    if not prompt_sent and prompt.startswith(token):
                        continue
                    prompt_sent = True
                    token_count += 1

                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=token))],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

            # Final chunk with finish_reason.
            final_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(), finish_reason="stop")],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            t.set_status(200)
            metrics.observe_tokens(endpoint="chat_completions", token_count=token_count)

        except Exception as exc:
            logger.exception("Error in chat stream generation")
            t.set_status(500)

            error_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=f"Error: {type(exc).__name__}"))
                ],
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
