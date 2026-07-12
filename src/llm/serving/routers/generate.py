"""Text generation endpoints (``/generate``, ``/batch_generate``)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from llm.serving.auth import get_api_key
from llm.serving.config import ServingConfig
from llm.serving.errors import APIError, ErrorCode
from llm.serving.schemas import (
    BatchGenerationRequest,
    BatchGenerationResponse,
    GenerationRequest,
    GenerationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"])

# Module-level references are set by ``llm.serving.api`` during lifespan
# startup. Keeping them here (instead of in the router closure) preserves
# the same module-level shape as before, so any future process-level
# instrumentation (e.g. a /debug route) can reach them without
# dependency-injection refactors.
config: ServingConfig | None = None
generation_service = None  # ServingGenerationService; see llm.serving.generation_service
inference_semaphore: asyncio.Semaphore | None = None


def configure(config_: ServingConfig, generation_service_, semaphore_: asyncio.Semaphore) -> None:
    """Bind the module-level references.

    Called once during FastAPI lifespan startup. Importing this module
    without configuring first will yield None values and the endpoints
    will refuse to serve (the ``RuntimeError`` below is a programming
    error, not a runtime condition).
    """
    global config, generation_service, inference_semaphore
    config = config_
    generation_service = generation_service_
    inference_semaphore = semaphore_


def _require_generation_service():
    if generation_service is None:
        raise RuntimeError("Generation service not initialized")
    return generation_service


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
        return StreamingResponse(
            _stream_generator(request), media_type="text/event-stream"
        )

    try:
        async with asyncio.timeout(config_.request_timeout):
            async with inference_semaphore:
                generated_text = await run_in_threadpool(
                    _sync_generate,
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                )
        return GenerationResponse(generated_text=generated_text, token_count=len(generated_text))
    except TimeoutError as exc:
        raise APIError(ErrorCode.TIMEOUT, "Request timeout") from exc
    except RuntimeError as exc:
        raise APIError(ErrorCode.MODEL_UNAVAILABLE, str(exc)) from exc
    except ValueError as exc:
        raise APIError(
            ErrorCode.INVALID_REQUEST, f"Invalid request: {exc}", details={"field": str(exc)}
        ) from exc
    except APIError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in generate_text")
        raise APIError(ErrorCode.INTERNAL, "Internal server error") from exc


async def _stream_generator(request: GenerationRequest) -> AsyncGenerator[str]:
    """Stream tokens from the sync engine as an SSE-friendly async iterable."""
    try:
        from starlette.concurrency import iterate_in_threadpool

        iterator = _sync_stream_generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        async for chunk in iterate_in_threadpool(iterator):
            yield chunk
    except Exception as exc:
        logger.exception("Error in stream generation")
        yield f"Error: {type(exc).__name__}"


@router.post("/batch_generate", response_model=BatchGenerationResponse)
async def batch_generate_text(
    request: BatchGenerationRequest,
    config_: Annotated[ServingConfig, Depends(lambda: config)],
    _api_key: Annotated[str, Depends(get_api_key)],
) -> BatchGenerationResponse:
    """Generate text for a batch of prompts in one call."""
    try:
        async with asyncio.timeout(config_.request_timeout):
            async with inference_semaphore:
                results = await run_in_threadpool(
                    _sync_batch_generate,
                    prompts=request.prompts,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                )
        return BatchGenerationResponse(
            results=[
                GenerationResponse(generated_text=text, token_count=len(text))
                for text in results
            ]
        )
    except TimeoutError as exc:
        raise APIError(ErrorCode.TIMEOUT, "Request timeout") from exc
    except RuntimeError as exc:
        raise APIError(ErrorCode.MODEL_UNAVAILABLE, str(exc)) from exc
    except ValueError as exc:
        raise APIError(
            ErrorCode.INVALID_REQUEST, f"Invalid request: {exc}", details={"field": str(exc)}
        ) from exc
    except APIError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in batch_generate_text")
        raise APIError(ErrorCode.INTERNAL, "Internal server error") from exc
