import pytest
from fastapi.testclient import TestClient

from llm.serving.api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
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

        chunks = []
        for line in response.iter_lines():
            if line and line.startswith("data: "):
                chunks.append(line)

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
def test_bearer_token_auth():
    """Test Bearer token authentication."""
    from fastapi.testclient import TestClient

    from llm.serving.api import app, config

    original_key = config.api_key
    config.api_key = "test-secret-key"

    try:
        with TestClient(app) as c:
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
