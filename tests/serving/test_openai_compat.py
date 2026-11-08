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
@pytest.mark.slow
def test_chat_non_streaming_preserves_whitespace(monkeypatch, client):
    """The non-streaming chat completion must NOT strip the completion's own
    leading/trailing whitespace (the model may legitimately emit a leading
    newline) — matches the streaming path, which emits chunks verbatim, and
    OpenAI semantics."""
    # The route does generated_text[len(prompt):]; make the mock compute it.
    monkeypatch.setattr("llm.serving.routers.chat._sync_generate", lambda prompt, **kw: prompt + "  hi  ")

    payload = {"model": "llm", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 50}
    data = client.post("/v1/chat/completions", json=payload).json()
    assert data["choices"][0]["message"]["content"] == "  hi  "


@pytest.mark.slow
def test_chat_finish_reason_length_when_budget_exhausted(monkeypatch, client):
    """finish_reason must be ``"length"`` (OpenAI spec) when generation consumed
    the full ``max_tokens`` budget, not hard-coded ``"stop"``."""
    # 4 chars >= max_tokens=2 → length.
    monkeypatch.setattr("llm.serving.routers.chat._sync_generate", lambda prompt, **kw: prompt + "abcd")

    payload = {"model": "llm", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 2}
    data = client.post("/v1/chat/completions", json=payload).json()
    assert data["choices"][0]["finish_reason"] == "length"


@pytest.mark.slow
def test_chat_finish_reason_stop_when_natural_stop(monkeypatch, client):
    """A short completion well under max_tokens keeps finish_reason ``"stop"``."""
    # 1 char < max_tokens=10 → stop.
    monkeypatch.setattr("llm.serving.routers.chat._sync_generate", lambda prompt, **kw: prompt + "a")

    payload = {"model": "llm", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}
    data = client.post("/v1/chat/completions", json=payload).json()
    assert data["choices"][0]["finish_reason"] == "stop"


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


@pytest.mark.slow
def test_chat_completions_runtime_error_returns_503(client, monkeypatch):
    """Non-streaming chat: if the generation service raises RuntimeError,
    the endpoint maps it to HTTP 503 (model unavailable)."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.generate.side_effect = RuntimeError("model exploded")
    mock.stream.return_value = iter([])
    monkeypatch.setattr("llm.serving.routers.generate.generation_service", mock)

    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
        "stream": False,
    }

    response = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 503


@pytest.mark.slow
def test_chat_completions_value_error_returns_400(client, monkeypatch):
    """Non-streaming chat: ValueError maps to HTTP 400 (invalid request)."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.generate.side_effect = ValueError("bad input")
    mock.stream.return_value = iter([])
    monkeypatch.setattr("llm.serving.routers.generate.generation_service", mock)

    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
        "stream": False,
    }

    response = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 400


def _finish_reason_from_stream(client, monkeypatch, stream_tokens, max_tokens):
    """Drive a streaming chat request and return the final ``finish_reason``."""
    import json
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.stream.return_value = iter(stream_tokens)
    mock.generate.return_value = "ok"
    monkeypatch.setattr("llm.serving.routers.generate.generation_service", mock)

    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        finish = None
        for line in response.iter_lines():
            if not (line and line.startswith("data: ") and line != "data: [DONE]"):
                continue
            data = json.loads(line[6:])
            fr = data["choices"][0].get("finish_reason")
            if fr is not None:
                finish = fr
        return finish


@pytest.mark.slow
def test_chat_stream_finish_reason_length_when_budget_exhausted(client, monkeypatch):
    """Regression: streaming chat must report
    ``finish_reason="length"`` when the stream consumed the full
    ``max_tokens`` budget — not hard-code ``"stop"`` like the non-streaming
    sibling (which reports ``"length"`` for identical input)."""
    # Exactly max_tokens (3) tokens streamed → the budget was exhausted.
    result = _finish_reason_from_stream(client, monkeypatch, ["a", "b", "c"], max_tokens=3)
    assert result == "length"


@pytest.mark.slow
def test_chat_stream_finish_reason_stop_when_natural_stop(client, monkeypatch):
    """A short stream well under max_tokens keeps finish_reason ``"stop"``."""
    result = _finish_reason_from_stream(client, monkeypatch, ["a"], max_tokens=10)
    assert result == "stop"


@pytest.mark.slow
def test_chat_completions_stream_error_yields_error_chunk(client, monkeypatch):
    """Streaming chat: if the generation service raises, the stream emits an
    error chunk and [DONE] rather than crashing the response."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.stream.side_effect = RuntimeError("model exploded")
    mock.generate.return_value = "ok"
    monkeypatch.setattr("llm.serving.routers.generate.generation_service", mock)

    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
        "stream": True,
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        chunks = [line for line in response.iter_lines() if line and line.startswith("data: ")]
        # Should include the error chunk and [DONE].
        assert chunks[-1] == "data: [DONE]"
        assert any("Error:" in chunk for chunk in chunks)


@pytest.mark.slow
def test_chat_completions_stream_does_not_drop_prompt_prefix_token(client, monkeypatch):
    """Streaming chat must NOT strip generated tokens that happen to be a
    prefix of the rendered prompt.

    Regression for: the stream handler used to drop the leading
    streamed token whenever ``prompt.startswith(token)`` was true, assuming
    the backend echoes the prompt. Backends never echo the prompt on the
    streaming path (only ``generate()`` prepends it for the non-streaming
    path), so that strip could silently delete a legitimate first token.
    Here the first generated token ``"u"`` is a prefix of the rendered
    prompt ``"user: hi\nAssistant: "``; it must appear verbatim in the SSE
    stream.
    """
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.stream.return_value = iter(["u", "sually", " done"])
    mock.generate.return_value = "ok"
    monkeypatch.setattr("llm.serving.routers.generate.generation_service", mock)

    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 3,
        "stream": True,
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        streamed = []
        for line in response.iter_lines():
            if not (line and line.startswith("data: ")):
                continue
            if line == "data: [DONE]":
                continue
            import json

            data = json.loads(line[6:])
            content = data["choices"][0].get("delta", {}).get("content")
            if content is not None:
                streamed.append(content)
        assert streamed == ["u", "sually", " done"]


def test_stream_terminal_chunk_carries_a_finish_reason(client):
    """The last data chunk before ``[DONE]`` must report WHY generation ended.

    OpenAI's SSE contract puts ``finish_reason`` on a terminal chunk with an
    empty delta; clients read it to distinguish a natural stop from a truncated
    response. The existing streaming test only asserts that ``[DONE]`` is
    present, so a server that ended the stream with a bare content chunk would
    pass while every well-behaved client saw ``finish_reason: null``.
    """
    import json

    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
        "stream": True,
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        data_lines = [line for line in response.iter_lines() if line and line.startswith("data: ")]

    assert data_lines[-1] == "data: [DONE]", "the stream must terminate with [DONE]"
    body_lines = data_lines[:-1]
    assert body_lines, "the stream carried no data chunks"

    # Everything except the terminal chunk carries delta content.
    terminal = json.loads(body_lines[-1][len("data: ") :])
    finish_reason = terminal["choices"][0]["finish_reason"]
    assert finish_reason in {"stop", "length"}, f"terminal chunk finish_reason was {finish_reason!r}"
    assert terminal["choices"][0]["delta"].get("content") in (None, ""), (
        "the terminal chunk must carry an EMPTY delta, not more content"
    )

    # Every chunk shares one completion id (clients group the stream by it).
    ids = {json.loads(line[len("data: ") :])["id"] for line in body_lines}
    assert len(ids) == 1, f"chunks disagreed on the completion id: {sorted(ids)}"


class TestModelsEndpoint:
    """``GET /v1/models`` is the reachability probe OpenAI clients send first.

    Its absence meant the official SDK's ``client.models.list()`` failed with a
    bare 404, which some clients report as "server unreachable" before ever
    attempting a completion — even though ``/v1/chat/completions`` worked.
    """

    def test_models_endpoint_lists_one_model(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200

        body = response.json()
        assert body["object"] == "list", f"unexpected envelope: {body}"
        assert isinstance(body["data"], list)
        assert len(body["data"]) == 1, "this server hosts exactly one model"

        entry = body["data"][0]
        assert entry["object"] == "model"
        assert entry["id"], "the model entry needs a non-empty id"
        assert "owned_by" in entry

    def test_models_endpoint_is_registered_before_any_auth_gate(self, client):
        """The probe must answer even when the client never sends a request body.

        A model listing that required an auth header would defeat the purpose:
        clients call it to decide whether the server is reachable at all.
        """
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert response.json()["data"][0]["id"]
