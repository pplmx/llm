"""FastAPI application factory for the serving API.

This module wires the pieces together:

- :class:`RequestIDMiddleware` (request correlation + access log)
- exception handlers that map every failure to the structured envelope
  in :mod:`llm.serving.errors`
- the three routers (health, generate, chat)
- :class:`prometheus_fastapi_instrumentator.Instrumentator`

All endpoint logic lives in the routers; this file stays focused on
process wiring (logging setup, lifespan, app construction) so adding a
new endpoint doesn't grow it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pythonjsonlogger import json

from llm.serving.auth import (
    api_key_header,
    authorization_header,
    get_api_key,
    is_loopback,
)
from llm.serving.auth import is_loopback as _is_loopback
from llm.serving.batch_engine import ContinuousBatchingEngine
from llm.serving.chat_template import (
    DEFAULT_CHAT_GENERATION_PREFIX,
    DEFAULT_CHAT_MESSAGE_TEMPLATE,
)
from llm.serving.chat_template import (
    messages_to_prompt as _messages_to_prompt,
)
from llm.serving.config import ServingConfig
from llm.serving.errors import register_exception_handlers
from llm.serving.generation_service import ServingGenerationService
from llm.serving.middleware import RequestIDMiddleware
from llm.serving.routers import chat as chat_router
from llm.serving.routers import generate as generate_router
from llm.serving.routers import health as health_router

# Backwards-compat re-exports. The auth, chat-template, middleware, and
# CLI helpers moved out of ``llm.serving.api`` during T2 #16, but their
# previous names remain importable from this module so existing call
# sites (and tests) keep working.
__all__ = [
    "DEFAULT_CHAT_GENERATION_PREFIX",
    "DEFAULT_CHAT_MESSAGE_TEMPLATE",
    "_is_loopback",
    "_messages_to_prompt",
    "api_key_header",
    "app",
    "authorization_header",
    "config",
    "get_api_key",
    "is_loopback",
]


def main() -> None:
    """Backwards-compat entry point.

    Real implementation lives in :mod:`llm.serving.cli`. Kept here as a
    thin alias so ``python -m llm.serving.api`` and ``api.main()`` keep
    working.
    """
    from llm.serving.cli import main as _cli_main

    _cli_main(config=config)


logger = logging.getLogger()
_log_handler = logging.StreamHandler(sys.stdout)
_formatter = json.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
_log_handler.setFormatter(_formatter)
logger.addHandler(_log_handler)

config = ServingConfig()
logger.setLevel(config.log_level)

# Concurrency control: bound concurrent in-flight generations regardless
# of FastAPI's worker count.
inference_semaphore = asyncio.Semaphore(config.max_concurrent_requests)


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, Any]:
    """FastAPI lifespan manager — load model, wire routers, log config."""
    logger.info("Starting up...")

    generation_service = ServingGenerationService.from_config(config)
    engine = ContinuousBatchingEngine.from_serving_config(
        config,
        model=generation_service.model,
        tokenizer=generation_service.tokenizer,
    )

    # Inject dependencies into the routers. Done here (not at import time)
    # so unit tests can replace them via ``monkeypatch.setattr`` on the
    # module-level attributes without instantiating a real model.
    generate_router.configure(config, generation_service, inference_semaphore)
    chat_router.configure(config, inference_semaphore)

    _log_server_config(generation_service, config)

    try:
        yield
    finally:
        engine.unload_model()
        logger.info("Shutting down...")


def _log_server_config(
    generation_service: ServingGenerationService,
    config: ServingConfig,
) -> None:
    """Emit a single structured JSON line describing the running server.

    Operators rely on this for incident triage: which model is loaded, on
    which device, with which attention impl, with prefix cache on/off,
    etc. Never log the api_key value itself — only whether it is set.
    """
    from llm.utils.common import count_parameters

    model = generation_service.model
    total_params, trainable_params = count_parameters(model)
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

# Middleware (order matters: outer middleware runs first on the way in,
# last on the way out — so the request_id is set before anything else
# reads it).
app.add_middleware(RequestIDMiddleware)

# Routers.
app.include_router(health_router.router)
app.include_router(generate_router.router)
app.include_router(chat_router.router)

# Prometheus RED metrics + custom /metrics endpoint.
Instrumentator().instrument(app).expose(app)


# Centralized error handling — see :func:`llm.serving.errors.register_exception_handlers`.
register_exception_handlers(app, logger=logger)


if __name__ == "__main__":
    # Lazy import so `python -m llm.serving.api` works without
    # pre-loading uvicorn at import time.
    from llm.serving.cli import main

    main()
