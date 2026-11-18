"""Shared fixtures for the serving test suite.

``client`` used to live in ``test_api.py``; the Responses API tests need the
same TestClient, so it is promoted here instead of duplicated.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm.serving.api import app


@pytest.fixture
def client(monkeypatch, device):
    """TestClient backed by a real tiny model (GPU-first, falls back to CPU).

    Used by @pytest.mark.slow tests that assert on actual generation output.
    Constructs the model on the session-scoped ``device`` (GPU when available)
    instead of forcing CPU.
    """
    from unittest.mock import MagicMock

    import torch

    from llm.generation.backends import EagerGenerationBackend
    from llm.models.decoder import DecoderModel
    from llm.serving.auth import api_key_header
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.config import ServingConfig
    from llm.serving.generation_service import ServingGenerationService

    torch.manual_seed(42)
    tiny_model = DecoderModel(
        vocab_size=100,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
        device=device,
    )
    from tests.support.tokenizers import StubTokenizer

    tokenizer = StubTokenizer()

    real_service = ServingGenerationService(
        model=tiny_model,
        tokenizer=tokenizer,
        backend=EagerGenerationBackend(),
        device=device,
    )

    fake_engine = MagicMock()
    monkeypatch.setattr(
        ServingGenerationService,
        "from_config",
        classmethod(lambda cls, config, **kw: real_service),
    )
    monkeypatch.setattr(
        ContinuousBatchingEngine,
        "from_serving_config",
        classmethod(lambda cls, config, **kw: fake_engine),
    )
    monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

    cfg = ServingConfig(
        api_key="test-key",
        request_timeout=30.0,
        device=str(device),
        generation_backend="eager",
    )

    with TestClient(app) as c:
        monkeypatch.setattr("llm.serving.routers.generate.generation_service", real_service)
        monkeypatch.setattr("llm.serving.routers.generate.config", cfg)
        monkeypatch.setattr("llm.serving.routers.chat.config", cfg)
        # The Responses endpoint reads its own module-level config reference
        # (bound in lifespan startup); without this it falls back to None and
        # the dependency raises.
        monkeypatch.setattr("llm.serving.routers.responses.config", cfg)
        c.headers[api_key_header.model.name] = "test-key"
        yield c
