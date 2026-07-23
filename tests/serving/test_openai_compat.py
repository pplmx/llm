import pytest
from fastapi.testclient import TestClient

from llm.serving.api import app


@pytest.fixture
def client(monkeypatch):
    from unittest.mock import MagicMock

    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    mock = MagicMock()
    mock.generate.return_value = "ok"
    mock.stream.return_value = iter([])
    fake_service = MagicMock()
    fake_engine = MagicMock()
    monkeypatch.setattr(ServingGenerationService, "from_config", classmethod(lambda cls, config, **kw: fake_service))
    monkeypatch.setattr(
        ContinuousBatchingEngine, "from_serving_config", classmethod(lambda cls, config, **kw: fake_engine)
    )
    monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)
    with TestClient(app) as c:
        monkeypatch.setattr("llm.serving.routers.generate.generation_service", mock)
        yield c


@pytest.mark.slow
def test_chat_completions_basic(client):
    """Test basic chat completion request."""
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
        "temperature": 0.5,
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert "id" in data
    assert "created" in data
    assert "choices" in data
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "content" in data["choices"][0]["message"]
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "usage" in data
    assert "prompt_tokens" in data["usage"]
    assert "completion_tokens" in data["usage"]
    assert "total_tokens" in data["usage"]


@pytest.mark.slow
def test_chat_completions_with_system_message(client):
    """Test chat completion with system message."""
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ],
        "max_tokens": 5,
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["choices"]) == 1


@pytest.mark.slow
def test_chat_completions_multi_turn(client):
    """Test multi-turn conversation."""
    payload = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you"},
        ],
        "max_tokens": 5,
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.slow
def test_chat_completions_stream(client):
    """Test streaming chat completion."""
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
        "stream": True,
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        chunks = [line for line in response.iter_lines() if line and line.startswith("data: ")]

        # Should have at least: first chunk (role), content chunks, final chunk, [DONE]
        assert len(chunks) >= 2
        assert chunks[-1] == "data: [DONE]"


@pytest.mark.slow
def test_chat_completions_empty_messages(client):
    """Test that empty messages list is rejected."""
    payload = {"messages": [], "max_tokens": 10}

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 422


@pytest.mark.slow
def test_chat_completions_invalid_role(client):
    """Test that invalid role is rejected."""
    payload = {
        "messages": [{"role": "invalid", "content": "hello"}],
        "max_tokens": 10,
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 422


@pytest.mark.slow
def test_bearer_token_auth(monkeypatch):
    """Test Bearer token authentication."""
    from unittest.mock import MagicMock

    import llm.serving.routers.generate as generate_module
    from llm.serving.api import app, config
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    original_key = config.api_key
    config.api_key = "test-secret-key"

    try:
        mock = MagicMock()
        mock.generate.return_value = "ok"
        monkeypatch.setattr(ServingGenerationService, "from_config", classmethod(lambda cls, config, **kw: MagicMock()))
        monkeypatch.setattr(
            ContinuousBatchingEngine, "from_serving_config", classmethod(lambda cls, config, **kw: MagicMock())
        )
        monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

        with TestClient(app) as c:
            monkeypatch.setattr(generate_module, "generation_service", mock)
            payload = {"messages": [{"role": "user", "content": "hi"}]}

            # No auth -> 403
            response = c.post("/v1/chat/completions", json=payload)
            assert response.status_code == 403

            # Wrong Bearer token -> 403
            response = c.post(
                "/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert response.status_code == 403

            # Correct Bearer token -> 200
            response = c.post(
                "/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer test-secret-key"},
            )
            assert response.status_code == 200

            # X-API-Key also works
            response = c.post(
                "/v1/chat/completions",
                json=payload,
                headers={"X-API-Key": "test-secret-key"},
            )
            assert response.status_code == 200
    finally:
        config.api_key = original_key


@pytest.mark.slow
def test_chat_completions_with_penalties(client):
    """Test presence_penalty mapping."""
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 5,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
