import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from pythonjsonlogger import json
from starlette.status import HTTP_403_FORBIDDEN

from llm.serving.config import ServingConfig
from llm.serving.engine import LLMEngine
from llm.serving.schemas import (
    BatchGenerationRequest,
    BatchGenerationResponse,
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    GenerationRequest,
    GenerationResponse,
)

logger = logging.getLogger()
logHandler = logging.StreamHandler(sys.stdout)
formatter = json.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

config = ServingConfig()
logger.setLevel(config.log_level)

engine = LLMEngine(config)

# Concurrency control
inference_semaphore = asyncio.Semaphore(config.max_concurrent_requests)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
authorization_header = APIKeyHeader(name="Authorization", auto_error=False)


def _extract_bearer_token(auth_header: str | None) -> str | None:
    """Extract token from Bearer authorization header."""
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


async def get_api_key(
    api_key_header: str | None = Security(api_key_header),
    auth_header: str | None = Security(authorization_header),
):
    """Verify API Key (supports X-API-Key and Bearer token)."""
    if not config.api_key:
        return None

    # Check X-API-Key header first
    if api_key_header == config.api_key:
        return api_key_header

    # Check Bearer token
    bearer_token = _extract_bearer_token(auth_header)
    if bearer_token == config.api_key:
        return bearer_token

    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials")


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, Any]:
    """
    FastAPI lifespan manager.
    """
    logger.info("Starting up...")
    engine.load_model()
    yield
    engine.unload_model()
    logger.info("Shutting down...")


app = FastAPI(
    title="LLM Inference API",
    description="Simple REST API for LLM text generation.",
    version="0.1.0",
    lifespan=lifespan,
)

# Prometheus Metrics
Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest, _api_key: str = Depends(get_api_key)
) -> GenerationResponse | StreamingResponse:
    """
    Text generation endpoint. Supports streaming and non-streaming.
    """
    if request.stream:
        return StreamingResponse(_stream_generator(request), media_type="text/event-stream")

    try:
        async with asyncio.timeout(config.request_timeout):
            async with inference_semaphore:
                generated_text = await run_in_threadpool(
                    engine.generate,
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                )

        return GenerationResponse(
            generated_text=generated_text,
            token_count=len(generated_text),
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_generator(request: GenerationRequest) -> AsyncGenerator[str]:
    """Helper generator to convert sync engine generation to async stream."""
    try:
        # Use iterate_in_threadpool (Starlette/FastAPI util) for streaming
        from starlette.concurrency import iterate_in_threadpool

        iterator = engine.stream_generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )

        async for chunk in iterate_in_threadpool(iterator):
            yield chunk

    except Exception as e:
        yield f"Error: {str(e)}"


@app.post("/batch_generate", response_model=BatchGenerationResponse)
async def batch_generate_text(
    request: BatchGenerationRequest, _api_key: str = Depends(get_api_key)
) -> BatchGenerationResponse:
    """
    Batch text generation endpoint for multiple prompts.
    """
    try:
        async with asyncio.timeout(config.request_timeout):
            async with inference_semaphore:
                results = await run_in_threadpool(
                    engine.batch_generate,
                    prompts=request.prompts,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                )

        return BatchGenerationResponse(
            results=[GenerationResponse(generated_text=text, token_count=len(text)) for text in results]
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# OpenAI-compatible Chat Completions API


def _messages_to_prompt(messages: list) -> str:
    """Convert chat messages to a simple prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"[System]: {msg.content}")
        elif msg.role == "user":
            parts.append(f"[User]: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"[Assistant]: {msg.content}")
    parts.append("[Assistant]:")
    return "\n".join(parts)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest, _api_key: str = Depends(get_api_key)
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    prompt = _messages_to_prompt(request.messages)
    repetition_penalty = 1.0 + request.presence_penalty

    if request.stream:
        return StreamingResponse(
            _chat_stream_generator(request, prompt, repetition_penalty),
            media_type="text/event-stream",
        )

    try:
        async with asyncio.timeout(config.request_timeout):
            async with inference_semaphore:
                generated_text = await run_in_threadpool(
                    engine.generate,
                    prompt=prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=repetition_penalty,
                )

        # Remove the prompt prefix from generated text
        if generated_text.startswith(prompt):
            completion = generated_text[len(prompt) :].strip()
        else:
            completion = generated_text.strip()

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
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _chat_stream_generator(
    request: ChatCompletionRequest, prompt: str, repetition_penalty: float
) -> AsyncGenerator[str]:
    """Generate SSE stream for chat completions."""
    import time
    import uuid

    from starlette.concurrency import iterate_in_threadpool

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    try:
        # First chunk with role
        first_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(role="assistant"))],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        iterator = engine.stream_generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=repetition_penalty,
        )

        prompt_sent = False
        async for token in iterate_in_threadpool(iterator):
            # Skip prompt tokens in streaming output
            if not prompt_sent and prompt.startswith(token):
                continue
            prompt_sent = True

            chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=token))],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(), finish_reason="stop")],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=f"Error: {e}"))],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


def main():
    """Entry point for llm-serve CLI."""
    import uvicorn

    uvicorn.run("llm.serving.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
