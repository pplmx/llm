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


async def get_api_key(
    api_key_header: str = Security(api_key_header),
):
    """Verify API Key."""
    if config.api_key:
        if api_key_header == config.api_key:
            return api_key_header
        else:
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials")
    return None


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("llm.serving.api:app", host="0.0.0.0", port=8000, reload=True)
