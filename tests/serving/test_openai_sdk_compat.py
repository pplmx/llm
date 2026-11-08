"""Validate the serving API against the OpenAI SDK's OWN response models.

The rest of the serving suite asserts the shapes we *intend* to emit — hand-written
``data["choices"][0]["text"]`` style checks. Those cannot catch a subtle contract
drift, because the expectation is written from the same understanding as the
implementation: rename ``object`` to ``text_completion_v1`` and the assertion
travels with it.

The OpenAI SDK ships pydantic models for every response type. Validating against
those makes an *external* party the authority: if ``Completion.model_validate``
accepts the body, a real client built on this SDK can parse it. That is the
property the ``/v1/completions`` and ``/v1/chat/completions`` endpoints exist to
provide, and nothing tested it.

Measured before these tests existed, against a live ``llm serve``: both
endpoints' bodies validate, so they are compatible today. The value is in
keeping it that way.

The SDK's *HTTP client* is not used (its bundled httpx shim mis-parses the
test-server URL in this environment — ``InvalidURL: Invalid port``); the
requests go through ``httpx`` directly and only the response MODELS are used.
That keeps the assertion on the wire format, which is the part that matters.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from llm.serving.api import app

openai = pytest.importorskip("openai", reason="openai SDK is an optional dependency")


@pytest.fixture
def client(monkeypatch):
    """TestClient with a mocked generation service (no real model loaded)."""
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    fake_service = MagicMock()
    fake_service.tokenizer.encode = lambda text: list(range(len(text)))
    fake_service.tokenizer.decode = lambda ids: "x"
    fake_service.generate.side_effect = lambda prompt, **kw: prompt + " hello world"
    fake_service.batch_generate.side_effect = lambda prompts, **kw: [p + " hello world" for p in prompts]
    fake_service.stream.return_value = iter(["hi"])

    monkeypatch.setattr(
        ServingGenerationService,
        "from_config",
        classmethod(lambda cls, config, **kw: fake_service),
    )
    monkeypatch.setattr(
        ContinuousBatchingEngine,
        "from_serving_config",
        classmethod(lambda cls, config, **kw: MagicMock()),
    )
    monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

    with TestClient(app) as c:
        monkeypatch.setattr("llm.serving.routers.generate.generation_service", fake_service)
        yield c


def test_completions_response_parses_as_an_sdk_completion(client):
    """``POST /v1/completions`` → ``openai.types.completion.Completion``.

    The legacy completions endpoint is consumed through the SDK's
    ``client.completions.create``, which parses the body with this model. A
    body that only *looks* right (``object`` misspelled, ``text`` missing,
    ``usage`` absent) raises there, in the client's own process, far from the
    server that produced it.
    """
    from openai.types.completion import Completion

    response = client.post("/v1/completions", json={"model": "llm", "prompt": "hi", "max_tokens": 5})
    assert response.status_code == 200, response.text

    parsed = Completion.model_validate(response.json())
    assert parsed.object == "text_completion"
    assert parsed.choices
    assert isinstance(parsed.choices[0].text, str)
    assert parsed.choices[0].finish_reason in ("stop", "length")
    assert parsed.usage is not None
    assert parsed.usage.total_tokens == parsed.usage.prompt_tokens + parsed.usage.completion_tokens


def test_completions_batch_response_parses_as_an_sdk_completion(client):
    """A list ``prompt`` yields a body the SDK model still accepts."""
    from openai.types.completion import Completion

    response = client.post("/v1/completions", json={"model": "llm", "prompt": ["a", "bb"], "max_tokens": 5})
    assert response.status_code == 200, response.text

    parsed = Completion.model_validate(response.json())
    assert [c.index for c in parsed.choices] == [0, 1]


def test_chat_response_parses_as_an_sdk_chat_completion(client):
    """``POST /v1/chat/completions`` → ``openai.types.chat.ChatCompletion``."""
    from openai.types.chat.chat_completion import ChatCompletion

    response = client.post(
        "/v1/chat/completions",
        json={"model": "llm", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
    )
    assert response.status_code == 200, response.text

    parsed = ChatCompletion.model_validate(response.json())
    assert parsed.object == "chat.completion"
    assert parsed.choices[0].message.role == "assistant"
    assert isinstance(parsed.choices[0].message.content, str)


def test_models_list_parses_as_the_sdk_model_page(client):
    """``GET /v1/models`` → the SDK's ``SyncPage[Model]`` item shape.

    ``client.models.list()`` is the reachability probe OpenAI clients make
    first, and it parses each entry with ``openai.types.model.Model``. A list
    whose entry lacks ``id``/``created``/``owned_by`` fails there even though
    the HTTP status is 200.
    """
    from openai.types.model import Model

    response = client.get("/v1/models")
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["object"] == "list"
    assert body["data"], "the models page must list at least the served model"
    for entry in body["data"]:
        parsed = Model.model_validate(entry)
        assert parsed.id


def test_streaming_chat_chunks_parse_as_sdk_chunk_models(client):
    """``stream=True`` frames must each be a valid ``ChatCompletionChunk``.

    Streaming is the mode an SDK user is most likely to hit and the hardest to
    eyeball: every ``data:`` frame is parsed independently, so one malformed
    chunk breaks the iterator mid-response.
    """
    import json

    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

    frames = []
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "llm", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "stream": True},
    ) as response:
        assert response.status_code == 200
        frames.extend(
            json.loads(line[6:])
            for line in response.iter_lines()
            if line and line.startswith("data: ") and line != "data: [DONE]"
        )

    assert frames, "the stream produced no data frames"
    for frame in frames:
        parsed = ChatCompletionChunk.model_validate(frame)
        assert parsed.object == "chat.completion.chunk"
