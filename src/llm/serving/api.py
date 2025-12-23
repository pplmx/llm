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
from llm.serving.schemas import GenerationRequest, GenerationResponse

# 配置结构化日志
logger = logging.getLogger()
logHandler = logging.StreamHandler(sys.stdout)
formatter = json.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# 全局推理引擎
config = ServingConfig()
# 更新日志级别
logger.setLevel(config.log_level)

engine = LLMEngine(config)

# API Key Security Scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(
    api_key_header: str = Security(api_key_header),
):
    """验证 API Key."""
    if config.api_key:
        if api_key_header == config.api_key:
            return api_key_header
        else:
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials")
    return None


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, Any]:
    """
    FastAPI 生命周期管理器.
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
    """健康检查端点."""
    return {"status": "ok"}


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest, _api_key: str = Depends(get_api_key)
) -> GenerationResponse | StreamingResponse:
    """
    文本生成端点. 支持流式和非流式.
    """
    if request.stream:
        return StreamingResponse(_stream_generator(request), media_type="text/event-stream")

    try:
        # 在线程池中运行以避免阻塞事件循环
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
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_generator(request: GenerationRequest) -> AsyncGenerator[str]:
    """辅助生成器, 将同步的 engine 生成转换为异步流."""
    try:
        # 在线程池中运行生成器
        # 注意: 由于 iterate_in_threadpool 不直接支持带参数的生成器调用
        # 我们需要手动包装或使用 run_in_threadpool 获取迭代器, 然后在 async for 中遍历

        # 简单实现: 我们在线程中获取完整的迭代器, 然后 wrap 成 async
        # 但这样会阻塞直到生成开始. 更理想的是逐个 yield.

        # 使用 iterate_in_threadpool (Starlette/FastAPI util)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("llm.serving.api:app", host="0.0.0.0", port=8000, reload=True)
