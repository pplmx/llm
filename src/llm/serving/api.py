import asyncio
import contextlib
import hmac
import logging
import sys
from collections.abc import AsyncGenerator
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from pythonjsonlogger import json
from starlette.status import HTTP_403_FORBIDDEN

from llm.serving.batch_engine import ContinuousBatchingEngine
from llm.serving.config import ServingConfig
from llm.serving.generation_service import ServingGenerationService
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
log_handler = logging.StreamHandler(sys.stdout)
formatter = json.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

config = ServingConfig()
logger.setLevel(config.log_level)

# Global references, initialized in lifespan
engine: ContinuousBatchingEngine | None = None
generation_service: ServingGenerationService | None = None

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
    """Verify API Key (supports X-API-Key and Bearer token).

    Comparison uses ``hmac.compare_digest`` to avoid leaking key bytes via timing.
    """
    if not config.api_key:
        return None

    expected = config.api_key
    # Check X-API-Key header first
    if api_key_header is not None and hmac.compare_digest(api_key_header, expected):
        return api_key_header

    # Check Bearer token
    bearer_token = _extract_bearer_token(auth_header)
    if bearer_token is not None and hmac.compare_digest(bearer_token, expected):
        return bearer_token

    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials")


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, Any]:
    """FastAPI lifespan manager."""
    global engine, generation_service
    logger.info("Starting up...")

    generation_service = ServingGenerationService.from_config(config)
    engine = ContinuousBatchingEngine.from_serving_config(
        config,
        model=generation_service.model,
        tokenizer=generation_service.tokenizer,
    )

    _log_server_config(generation_service, config)

    yield

    if engine:
        engine.unload_model()
    logger.info("Shutting down...")


def _log_server_config(
    generation_service: ServingGenerationService,
    config: ServingConfig,
) -> None:
    """Emit a single structured JSON line describing the running server.

    Operators rely on this for incident triage: which model is loaded, on which
    device, with which attention impl, with prefix cache on/off, etc. Never
    log the api_key value itself — only whether it is set.
    """
    from llm.utils.common import count_parameters

    model = generation_service.model
    total_params, trainable_params = count_parameters(model)
    # Discover dtype + device from the first parameter — robust to whichever
    # backend (DDP wrapper, FSDP, raw model) the engine handed us.
    try:
        first_param = next(model.parameters())
        dtype_str = str(first_param.dtype)
        device_str = str(first_param.device)
    except StopIteration:
        dtype_str = "unknown"
        device_str = "unknown"

    logger.info(
        "server_config",
        extra={
            "event": "server_config",
            "model_class": type(model).__name__,
            "param_count_total": total_params,
            "param_count_trainable": trainable_params,
            "dtype": dtype_str,
            "device": device_str,
            "max_seq_len": config.max_seq_len,
            "attn_impl": config.attn_impl,
            "mlp_impl": config.mlp_impl,
            "generation_backend": config.generation_backend,
            "enable_prefix_cache": config.enable_prefix_cache,
            "use_paged_attention": config.use_paged_attention,
            "api_key_set": bool(config.api_key),
        },
    )


app = FastAPI(
    title="LLM Inference API",
    description="Simple REST API for LLM text generation.",
    version="0.1.0",
    lifespan=lifespan,
)

# Prometheus Metrics
Instrumentator().instrument(app).expose(app)


# --- Generation Helper Functions ---


def _require_generation_service() -> ServingGenerationService:
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


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest, _api_key: Annotated[str, Depends(get_api_key)]
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
                    _sync_generate,
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e!s}")
    except Exception:
        # Log unexpected errors for debugging but return generic message
        import logging

        logger = logging.getLogger(__name__)
        logger.exception("Unexpected error in generate_text")
        raise HTTPException(status_code=500, detail="Internal server error")


async def _stream_generator(request: GenerationRequest) -> AsyncGenerator[str]:
    """Helper generator to convert sync engine generation to async stream."""
    try:
        # Use iterate_in_threadpool (Starlette/FastAPI util) for streaming
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

    except Exception as e:
        # Log the error but don't expose internal details in stream
        import logging

        logger = logging.getLogger(__name__)
        logger.exception("Error in stream generation")
        yield f"Error: {type(e).__name__}"


@app.post("/batch_generate", response_model=BatchGenerationResponse)
async def batch_generate_text(
    request: BatchGenerationRequest, _api_key: Annotated[str, Depends(get_api_key)]
) -> BatchGenerationResponse:
    """
    Batch text generation endpoint for multiple prompts.
    """
    try:
        async with asyncio.timeout(config.request_timeout):
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
            results=[GenerationResponse(generated_text=text, token_count=len(text)) for text in results]
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e!s}")
    except Exception:
        # Log unexpected errors for debugging
        import logging

        logger = logging.getLogger(__name__)
        logger.exception("Unexpected error in batch_generate_text")
        raise HTTPException(status_code=500, detail="Internal server error")


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
    request: ChatCompletionRequest, _api_key: Annotated[str, Depends(get_api_key)]
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
                    _sync_generate,
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e!s}")
    except Exception:
        # Log unexpected errors for debugging
        import logging

        logger = logging.getLogger(__name__)
        logger.exception("Unexpected error in chat_completions")
        raise HTTPException(status_code=500, detail="Internal server error")


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

        iterator = _sync_stream_generate(
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
        # Log the error for debugging
        import logging

        logger = logging.getLogger(__name__)
        logger.exception("Error in chat stream generation")

        error_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=f"Error: {type(e).__name__}"))],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


def _is_loopback(host: str) -> bool:
    """Return True if ``host`` is a loopback address.

    Covers the common cases: ``127.0.0.0/8`` and ``::1``. Anything else
    (``0.0.0.0``, ``*``, LAN IPs, public hostnames) is treated as non-loopback.
    """
    if host in ("127.0.0.1", "localhost", "::1"):
        return True
    return bool(host.startswith("127."))


def main():
    """Entry point for llm-serve CLI.

    ``reload=True`` is intentionally disabled because uvicorn's file-watcher
    conflicts with ``from llm.serving.api import app`` (the watch import path is
    incompatible with production use). For local development with auto-reload,
    run uvicorn directly::

        uvicorn llm.serving.api:app --reload --host 127.0.0.1 --port 8000

    Refuses to start when the server would bind to a non-loopback address
    without an ``api_key`` configured. ``host=0.0.0.0`` without auth exposes
    the inference endpoint to the network; this guard makes that mistake fail
    loudly at startup rather than silently at runtime.
    """
    import os

    import uvicorn

    if not _is_loopback(config.host) and not config.api_key:
        raise RuntimeError(
            f"Refusing to start: ServingConfig.host='{config.host}' binds to a "
            f"non-loopback address but api_key is not set. Anonymous access on a "
            f"public interface is unsafe. Either set LLM_SERVING_HOST to a loopback "
            f"address (127.0.0.1) or set LLM_SERVING_API_KEY."
        )

    reload = os.environ.get("LLM_SERVING_RELOAD", "").lower() in ("1", "true", "yes")
    uvicorn.run("llm.serving.api:app", host=config.host, port=8000, reload=reload)


if __name__ == "__main__":
    main()
